import math
import numpy as np

import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon import nn
from dgl import function as fn


def matmul_maybe_select(A, B):
    """Perform Matrix multiplication C = A * B but A could be an integer id vector.

    If A is an integer vector, we treat it as multiplying a one-hot encoded tensor.
    In this case, the expensive dense matrix multiply can be replaced by a much
    cheaper index lookup.

    For example,
    ::

        A = [2, 0, 1],
        B = [[0.1, 0.2],
             [0.3, 0.4],
             [0.5, 0.6]]

    then matmul_maybe_select(A, B) is equivalent to
    ::

        [[0, 0, 1],     [[0.1, 0.2],
         [1, 0, 0],  *   [0.3, 0.4],
         [0, 1, 0]]      [0.5, 0.6]]

    In all other cases, perform a normal matmul.

    Parameters
    ----------
    A : mxnet.NDArray
        lhs tensor
    B : mxnet.NDArray
        rhs tensor

    Returns
    -------
    C : mxnet.NDArray
        result tensor
    """
    if A.dtype in (np.int32, np.int64) and len(A.shape) == 1:
        return nd.take(B, A, axis=0)
    else:
        return nd.dot(A, B)


def bmm_maybe_select(A, B, index):
    """Slice submatrices of A by the given index and perform bmm.

    B is a 3D tensor of shape (N, D1, D2), which can be viewed as a stack of
    N matrices of shape (D1, D2). The input index is an integer vector of length M.
    A could be either:
    (1) a dense tensor of shape (M, D1),
    (2) an integer vector of length M.
    The result C is a 2D matrix of shape (M, D2)

    For case (1), C is computed by bmm:
    ::

        C[i, :] = matmul(A[i, :], B[index[i], :, :])

    For case (2), C is computed by index select:
    ::

        C[i, :] = B[index[i], A[i], :]

    Parameters
    ----------
    A : mxnet.NDArray
        lhs tensor
    B : mxnet.NDArray
        rhs tensor
    index : mxnet.NDArray
        index tensor

    Returns
    -------
    C : mxnet.NDArray
        return tensor
    """
    if A.dtype in (np.int32, np.int64) and len(A.shape) == 1:
        return B[index, A, :]
    else:
        BB = nd.take(B, index, axis=0)
        return nd.batch_dot(A.expand_dims(1), BB).squeeze(1)


class RelGraphConv(gluon.Block):
    r"""Relational graph convolution layer.

    Relational graph convolution is introduced in "`Modeling Relational Data with Graph
    Convolutional Networks <https://arxiv.org/abs/1703.06103>`__"
    and can be described as below:

    .. math::

      h_i^{(l+1)} = \sigma(\sum_{r\in\mathcal{R}}
      \sum_{j\in\mathcal{N}^r(i)}\frac{1}{c_{i,r}}W_r^{(l)}h_j^{(l)}+W_0^{(l)}h_i^{(l)})

    where :math:`\mathcal{N}^r(i)` is the neighbor set of node :math:`i` w.r.t. relation
    :math:`r`. :math:`c_{i,r}` is the normalizer equal
    to :math:`|\mathcal{N}^r(i)|`. :math:`\sigma` is an activation function. :math:`W_0`
    is the self-loop weight.

    The basis regularization decomposes :math:`W_r` by:

    .. math::

      W_r^{(l)} = \sum_{b=1}^B a_{rb}^{(l)}V_b^{(l)}

    where :math:`B` is the number of bases.

    The block-diagonal-decomposition regularization decomposes :math:`W_r` into :math:`B`
    number of block diagonal matrices. We refer :math:`B` as the number of bases.

    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    num_rels : int
        Number of relations.
    regularizer : str
        Which weight regularizer to use "basis" or "bdd"
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """
    def __init__(self,
                 in_feat,
                 out_feat,
                 num_rels,
                 regularizer="basis",
                 num_bases=None,
                 bias=True,
                 activation=None,
                 self_loop=False,
                 dropout=0.0):
        super(RelGraphConv, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.regularizer = regularizer
        self.num_bases = num_bases    # b in equal(3)
        if self.num_bases is None or self.num_bases > self.num_rels or self.num_bases < 0:
            self.num_bases = self.num_rels
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        if regularizer == "basis":
            # add basis weights  V in equal(3)
            self.weight = self.params.get(
                'weight', shape=(self.num_bases, self.in_feat, self.out_feat),
                init=mx.init.Xavier(magnitude=math.sqrt(2.0)))
            if self.num_bases < self.num_rels:
                # linear combination coefficients
                self.w_comp = self.params.get(
                    'w_comp', shape=(self.num_rels, self.num_bases),
                    init=mx.init.Xavier(magnitude=math.sqrt(2.0)))
            # message func
            self.message_func = self.basis_message_func
        elif regularizer == "bdd":
            if in_feat % num_bases != 0 or out_feat % num_bases != 0:
                raise ValueError('Feature size must be a multiplier of num_bases.')
            # add block diagonal weights
            self.submat_in = in_feat // self.num_bases
            self.submat_out = out_feat // self.num_bases

            # assuming in_feat and out_feat are both divisible by num_bases
            self.weight = self.params.get(
                'weight',
                shape=(self.num_rels, self.num_bases * self.submat_in * self.submat_out),
                init=mx.init.Xavier(magnitude=math.sqrt(2.0)))
            # message func
            self.message_func = self.bdd_message_func
        else:
            raise ValueError("Regularizer must be either 'basis' or 'bdd'")

        # bias
        if self.bias:
            self.h_bias = self.params.get('bias', shape=(out_feat,),
                                          init=mx.init.Zero())

        # weight for self loop
        if self.self_loop:
            self.loop_weight = self.params.get(
                'W_0', shape=(in_feat, out_feat),
                init=mx.init.Xavier(magnitude=math.sqrt(2.0)))

        self.dropout = nn.Dropout(dropout)

    def basis_message_func(self, edges):
        ctx = edges.src['h'].context
        if self.num_bases < self.num_rels:
            weight = self.weight.data(ctx).reshape(
                self.num_bases, self.in_feat * self.out_feat)
            weight = nd.dot(self.w_comp.data(ctx), weight).reshape(
                self.num_rels, self.in_feat, self.out_feat)
        else:
            weight = self.weight.data(ctx)
        msg = bmm_maybe_select(edges.src['h'], weight, edges.data['type'])
        # msg = utils.bmm_maybe_select(edges.src['h'], weight, edges.data['type'])
        if 'norm' in edges.data:
            msg = msg * edges.data['norm']
        return {'msg': msg}

    def bdd_message_func(self, edges):
        """Message function for block-diagonal-decomposition regularizer"""
        ctx = edges.src['h'].context
        if edges.src['h'].dtype in (np.int32, np.int64) and len(edges.src['h'].shape) == 1:
            raise TypeError('Block decomposition does not allow integer ID feature.')
        weight = self.weight.data(ctx)[edges.data['type'], :].reshape(
            -1, self.submat_in, self.submat_out)
        node = edges.src['h'].reshape(-1, 1, self.submat_in)
        msg = nd.batch_dot(node, weight).reshape(-1, self.out_feat)
        if 'norm' in edges.data:
            msg = msg * edges.data['norm']
        return {'msg': msg}

    def forward(self, g, x, etypes, norm=None):
        r"""Forward computation

        Parameters
        ----------
        g : DGLGraph
            The graph.
        x : mx.ndarray.NDArray
            Input node features. Could be either
              - :math:`(|V|, D)` dense tensor
              - :math:`(|V|,)` int64 vector, representing the categorical values of each
                node. We then treat the input feature as an one-hot encoding feature.
        etypes : mx.ndarray.NDArray
            Edge type tensor. Shape: :math:`(|E|,)`
        norm : mx.ndarray.NDArray
            Optional edge normalizer tensor. Shape: :math:`(|E|, 1)`

        Returns
        -------
        mx.ndarray.NDArray
            New node features.
        """
        with g.local_scope():
            g.ndata['h'] = x
            g.edata['type'] = etypes
            if norm is not None:
                g.edata['norm'] = norm
            if self.self_loop:
                loop_message = matmul_maybe_select(x, self.loop_weight.data(x.context))
                # loop_message = utils.matmul_maybe_select(x, self.loop_weight.data(x.context))

            # message passing
            g.update_all(self.message_func, fn.mean(msg='msg', out='h'))

            # apply bias and activation
            node_repr = g.ndata['h']
            if self.bias:
                node_repr = node_repr.reshape(node_repr.shape[0], -1) + self.h_bias.data(x.context)
            if self.self_loop:
                node_repr = node_repr + loop_message
            if self.activation:
                node_repr = self.activation(node_repr)
            node_repr = self.dropout(node_repr)
            return node_repr
