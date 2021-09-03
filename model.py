from mxnet import gluon
import mxnet as mx
from mxnet.gluon import nn, rnn
from mxnet import nd
from relgraphconv import *


class SARGCN(gluon.Block):
    def __init__(self, in_feat, h_dim, out_dim, num_rels, num_bases=-1,
                 num_hidden_layers=1, dropout=0.0,
                 use_self_loop=False, gpu_id=-1,
                 residual=False):
        super(SARGCN, self).__init__()
        self.in_feat = in_feat
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.gpu_id = gpu_id
        self.residual = residual
        if residual:
            self.res_fc = nn.Dense(out_dim)
        # create rgcn layers
        self.build_model()

    def build_model(self):
        self.layers = gluon.nn.Sequential()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.add(i2h)
        # h2h
        for idx in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer(idx)
            self.layers.add(h2h)
        # h2o
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.add(h2o)

    def build_input_layer(self):
        return SARGCNLayer(self.in_feat, self.h_dim, self.num_rels,
                           activation=mx.nd.relu,
                           dropout=self.dropout,
                           use_self_loop=self.use_self_loop)

    def build_hidden_layer(self, idx):
        return SARGCNLayer(self.h_dim, self.h_dim, self.num_rels,
                           activation=mx.nd.relu,
                           dropout=self.dropout,
                           use_self_loop=self.use_self_loop)


    def build_output_layer(self):
        return SARGCNLayer(self.h_dim, self.out_dim, self.num_rels,
                           activation=mx.nd.relu,
                           dropout=self.dropout,
                           use_self_loop=self.use_self_loop)


    def forward(self, g, h, r, norm):
        h0 = h
        for layer in self.layers:
            h = layer(g, h, r, norm)
        if self.residual:
            h = h+self.res_fc(h0)
        return h
    
    
class SARGCNLayer(nn.Block):
    def __init__(self, in_feats, out_feats, num_rels, groups=2,
                 radix=2, regularizer='basis', dropout=0.0,
                 activation=None, inter_channel=96, use_self_loop=False,
                 residual=True):
        super().__init__()
        self.radix = radix
        self.cardinality = groups
        self.G = groups * radix
        self.cardinal_in_feats = in_feats // groups // radix
        self.cardinal_out_feats = out_feats // groups // radix
        self.num_rels = num_rels
        self.regularizer = regularizer
        self.activation = activation
        self.dropout = dropout
        self.groups = groups
        self.radix = radix
        self.out_feats = out_feats
        self.use_self_loop = use_self_loop
        self.rgcn_list = []
        self.residual = residual
        for l in range(self.G):
            self.rgcn_list.append(RelGraphConv(self.cardinal_in_feats,
                                               out_feats // groups,
                                               self.num_rels,
                                               self.regularizer,
                                               activation=self.activation,
                                               self_loop=self.use_self_loop,
                                               dropout=self.dropout))
        for i, layer in enumerate(self.rgcn_list):
            self.register_child(layer, "gat_layer_{}".format(i))
        # soft attention, only one fc is utilized, while paper used 2.
        self.s2a1 = nn.Dense(inter_channel*self.groups,
                             activation='relu')
        self.s2a2 = nn.Dense(out_feats*radix,
                             activation='relu')
        self.LSTM = rnn.LSTM(self.out_feats//self.groups,
                             layout='NTC')
        self.out_fc = nn.Dense(self.out_feats,
                               activation='relu')
        if self.residual:
            self.res_fc = nn.Dense(out_feats)

    def feat_split(self, g, x):
        cardinal_graphs = []
        for i in range(self.G):
            _ = g.local_var()
            _.ndata.clear()
            _.ndata['h'] = x[:, self.cardinal_in_feats*i: self.cardinal_in_feats*(i+1)]
            cardinal_graphs.append(_)
        return cardinal_graphs

    def forward(self, g, x, etypes, norm=None):
        cardinal_groups = self.feat_split(g, x)
        U = list()
        for l in range(self.G):
            U.append(self.rgcn_list[l](cardinal_groups[l],
                                       cardinal_groups[l].ndata['h'],
                                       cardinal_groups[l].edata['type']))
        # Here, we want to obtain $b\in (6324, 2, 2, 20)$.
        # Hence, we utilzie the concat operation and reshape.
        U_0 = nd.stack(*U, axis=1)
        U = self.LSTM(U_0).reshape(-1,
                                 self.groups,
                                 self.radix,
                                 self.out_feats//self.groups)
        U_hat = U.sum(axis=2)    # $U_{hat} \in (batch*nodes, groups, out_feats//groups)$
        # average pooling
        s = U_hat.reshape(U_hat.shape[0]//80, 80, U_hat.shape[1], U_hat.shape[2]).mean(axis=1).reshape(-1, self.out_feats)
        # following Hang Zhang, we utilize two FC layers.
        a = nd.softmax(self.s2a2(self.s2a1(s)).reshape(s.shape[0], self.groups, self.radix, -1), axis=2).expand_dims(axis=1)
        V = nd.broadcast_mul(a, U.reshape(U.shape[0]//80, 80, U.shape[1], U.shape[2], U.shape[3])).sum(axis=3)
        out = self.out_fc(V.reshape(-1, self.out_feats))
        
        if self.residual:
            return out + self.res_fc(x)
        else:
            return out
