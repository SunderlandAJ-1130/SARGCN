import time
import numpy as np
import mxnet as mx
import os

os.environ['DGLBACKEND'] = 'mxnet'

import math
from mxnet import gluon, nd
from utils import collate, rmse_trainset, rmse_testset, GraphTraffic, \
    load_adjmatrix, EarlyStopping, train_seed, StandardScaler, masked_rmse_np, masked_mae_np, masked_mape_np
from model import SARGCN
import argparse
import warnings
import pickle

warnings.filterwarnings('ignore')

# def args
parser = argparse.ArgumentParser()
parser.add_argument('--seq_len', type=int, default=4)
parser.add_argument('--pre_len', type=int, default=4)
parser.add_argument('--num_node', type=int, default=80)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--n_epochs', type=int, default=500)
parser.add_argument('--n_bases', type=int, default=-1)
parser.add_argument('--num_rels', type=int, default=25)
parser.add_argument('--num_classes', type=int, default=8)
parser.add_argument('--dropout', type=float, default=0.05)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--n_layers', type=int, default=1)
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--test_batch_size', type=int, default=32)
parser.add_argument('--n_hidden', type=int, default=224)
parser.add_argument('--n_channel', type=int, default=96)
parser.add_argument('--train', type=str, default='../data/train.pkl')
parser.add_argument('--val', type=str, default='../data/val.pkl')
parser.add_argument('--test', type=str, default='../data/test.pkl')
parser.add_argument('--cn_adj_path', type=str, default='../data/adjacency_matrix_13.mat')
parser.add_argument('--device', type=int, default=1)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--edge_type', type=str, default='../data/edge_type_13.npy')
args = parser.parse_args()
print('Training configs: {}'.format(args))

# random seed
train_seed(seed=args.seed)
args.device = mx.gpu(args.device)


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, acc, model):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score > self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        model.save_parameters('SARGCN.param')


def load_data():
    # load data
    matfn = args.cn_adj_path
    g, tratim_matrix = load_adjmatrix(matfn)
    edge_type = np.load(args.edge_type)
    edge_type = mx.nd.array(edge_type, ctx=args.device)

    # load data
    with open(args.train, 'rb') as f:
        train = pickle.load(f)

    with open(args.val, 'rb') as f:
        val = pickle.load(f)

    with open(args.test, 'rb') as f:
        test = pickle.load(f)

    train_x, train_y = train['x'], train['y']
    valid_x, valid_y = val['x'], val['y']
    test_x, test_y = test['x'], test['y']

    train_x = np.swapaxes(train_x, 1, 2)
    train_x = np.concatenate([train_x[:, :, :, 0], train_x[:, :, :, 1]], axis=2)
    train_y = np.swapaxes(train_y, 1, 2)
    train_y = np.concatenate([train_y[:, :, :, 0], train_y[:, :, :, 1]], axis=2)

    valid_x = np.swapaxes(valid_x, 1, 2)
    valid_x = np.concatenate([valid_x[:, :, :, 0], valid_x[:, :, :, 1]], axis=2)
    valid_y = np.swapaxes(valid_y, 1, 2)
    valid_y = np.concatenate([valid_y[:, :, :, 0], valid_y[:, :, :, 1]], axis=2)

    test_x = np.swapaxes(test_x, 1, 2)
    test_x = np.concatenate([test_x[:, :, :, 0], test_x[:, :, :, 1]], axis=2)
    test_y = np.swapaxes(test_y, 1, 2)
    test_y = np.concatenate([test_y[:, :, :, 0], test_y[:, :, :, 1]], axis=2)

    scaler_axis = (0, 1, 2)
    scaler = StandardScaler(mean=train_x.mean(axis=scaler_axis),
                            std=train_x.std(axis=scaler_axis))
    train_x = nd.array(train_x, ctx=args.device)
    train_y = nd.array(train_y, ctx=args.device)
    valid_x = nd.array(valid_x, ctx=args.device)
    valid_y = nd.array(valid_y, ctx=args.device)
    test_x = nd.array(test_x, ctx=args.device)
    test_y = nd.array(test_y, ctx=args.device)

    train_x = scaler.transform(train_x)
    train_y = scaler.transform(train_y)
    valid_x = scaler.transform(valid_x)
    valid_y = scaler.transform(valid_y)
    test_x = scaler.transform(test_x)
    test_y = scaler.transform(test_y)

    num_train = train_x.shape[0]
    num_valid = valid_x.shape[0]
    num_test = test_x.shape[0]

    seq_len = args.seq_len
    pre_len = args.pre_len

    # divide training set, valid set and test set
    trainset = GraphTraffic(num_train, args.num_node, tratim_matrix)
    validset = GraphTraffic(num_valid, args.num_node, tratim_matrix)
    testset = GraphTraffic(num_test, args.num_node, tratim_matrix)

    for i in range(trainset.__len__()):
        trainset.graphs[i].ndata['h'] = train_x[i]
        trainset.graphs[i].edata['type'] = edge_type.reshape(-1, )
        trainset.labels[i] = train_y[i]
    for i in range(validset.__len__()):
        validset.graphs[i].ndata['h'] = valid_x[i]
        validset.graphs[i].edata['type'] = edge_type.reshape(-1, )
        validset.labels[i] = valid_y[i]
    for i in range(testset.__len__()):
        testset.graphs[i].ndata['h'] = test_x[i]
        testset.graphs[i].edata['type'] = edge_type.reshape(-1, )
        testset.labels[i] = test_y[i]

    return scaler, trainset, validset, testset


def main():
    scaler, trainset, validset, testset = load_data()
    dropout = args.dropout
    in_feats = 2 * args.seq_len
    num_classes = args.num_classes
    lr = args.lr
    weight_decay = 0.0
    n_bases = args.n_bases
    n_layers = args.n_layers
    n_epochs = args.n_epochs
    num_rels = args.num_rels
    use_self_loop = True
    batch_size = args.batch_size
    test_batch_size = args.test_batch_size
    n_hidden = args.n_hidden
    n_channel = args.n_channel

    # create model
    model = SARGCN(in_feats,
                   n_hidden,
                   num_classes,
                   num_rels,
                   inter_channel=n_channel,
                   num_bases=n_bases,
                   num_hidden_layers=n_layers,
                   dropout=dropout,
                   use_self_loop=use_self_loop,
                   gpu_id=args.device,
                   residual=False)
    model.initialize(mx.init.Xavier(magnitude=math.sqrt(2.0)), ctx=args.device)
    trainer = gluon.Trainer(model.collect_params(),
                            'adam',
                            {'learning_rate': lr,
                             'wd': weight_decay})
    loss = gluon.loss.L2Loss()
    stopper = EarlyStopping(patience=args.patience)
    mx.random.seed(args.seed)
    train_iter = gluon.data.DataLoader(trainset, batch_size, shuffle=True,
                                       batchify_fn=collate, last_batch='discard',
                                       thread_pool=True)
    valid_iter = gluon.data.DataLoader(validset, test_batch_size, shuffle=False,
                                       batchify_fn=collate, last_batch='discard',
                                       thread_pool=True)
    test_iter = gluon.data.DataLoader(testset, test_batch_size, shuffle=False,
                                      batchify_fn=collate, last_batch='discard',
                                      thread_pool=True)
    TrainLoss, ValidLoss, TestLoss = [], [], []
    TIME = []

    for epoch in range(n_epochs):
        train_l_sum, n = 0.0, 0
        start_time = time.time()
        for iter, (bg, label) in enumerate(train_iter):
            model.g = bg
            with mx.autograd.record():
                edge_type = bg.edata['type']
                pred = model(bg, bg.ndata['h'], edge_type, None)
                l = loss(pred, label)
            l.backward()
            trainer.step(batch_size)
            train_l_sum += l
            n += label.shape[0]
        train_l = train_l_sum
        train_rmse, train_loss, train_mae = rmse_trainset(model, train_iter, batch_size, args.num_node, scaler,
                                                          device=args.device)
        valid_rmse, valid_loss, _, _, _, _ = rmse_testset(model, valid_iter, test_batch_size, args.num_node, scaler,
                                                          mae=True, device=args.device)
        TIME.append(time.time()-start_time)
        test_rmse, test_loss, _, _, _, _ = rmse_testset(model, test_iter, test_batch_size, args.num_node, scaler,
                                                        mae=True, device=args.device)
        print(
            'epoch %d | running time %.2f s | train loss %.3f | valid loss %.3f | test loss %.3f | test rmse %.3f' %
            (epoch + 1, time.time() - start_time, train_loss, valid_loss, test_loss, test_rmse))
        TrainLoss.append(train_loss)
        ValidLoss.append(valid_loss)
        TestLoss.append(test_loss)
        if stopper.step(valid_rmse, model):
            break

    # evaluate
    model.load_parameters('SARGCN.param')
    test_rmse, test_mse, test_mae, test_mape, true, pred = rmse_testset(model, test_iter, batch_size, args.num_node,
                                                                        scaler, mae=True, device=args.device)
    for i in range(args.seq_len):
        rmse = masked_rmse_np(labels=true[:, :, [i, i + 4]], preds=pred[:, :, [i, i + 4]])
        mae = masked_mae_np(labels=true[:, :, [i, i + 4]], preds=pred[:, :, [i, i + 4]])
        mape = masked_mape_np(labels=true[:, :, [i, i + 4]], preds=pred[:, :, [i, i + 4]])
        print('time: %i, rmse: %.3f, mae: %.3f, mape: %.5f' % ((i + 1) * 15, rmse, mae, mape))

    model.save_parameters('result/SARGCN_RMSE_' + str(n_hidden) + '_' + str(n_channel) + '_' + 'ReLu_' + str(
        args.pre_len) + '_' + str(test_rmse) + '.param')
    TrainLoss = np.array(TrainLoss)
    ValidLoss = np.array(ValidLoss)
    TestLoss = np.array(TestLoss)
    TrainLoss = np.array(TrainLoss)
    ValidLoss = np.array(ValidLoss)
    TestLoss = np.array(TestLoss)
    np.save(
        'result/TrainLoss_RMSE_seed' + str(args.seed) + '_' + str(n_hidden) + '_' + str(n_channel) + '_' + 'ReLu_' + str(
            args.pre_len) + '_' + str(test_rmse) + '.npy', arr=TrainLoss)
    np.save(
        'result/ValidLoss_RMSE_seed' + str(args.seed) + '_' + str(n_hidden) + '_' + str(n_channel) + '_' + 'ReLu_' + str(
            args.pre_len) + '_' + str(test_rmse) + '.npy', arr=ValidLoss)
    np.save(
        'result/TestLoss_RMSE_seed' + str(args.seed) + '_' + str(n_hidden) + '_' + str(n_channel) + '_' + 'ReLu_' + str(
            args.pre_len) + '_' + str(test_rmse) + '.npy', arr=TestLoss)
    np.save('result/args' + str(test_rmse) + '.npy', arr=args)
    print(np.array(TIME).mean())


if __name__ == '__main__':
    main()
