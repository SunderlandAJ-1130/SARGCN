import numpy as np
import mxnet as mx
import dgl
from scipy import io as sio
import random


device = mx.gpu(0)

def masked_rmse_np(labels, preds, null_val=0):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels, null_val=null_val))


def masked_mse_np(labels, preds, null_val=0):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        rmse = np.square(np.subtract(preds, labels)).astype('float32')
        rmse = np.nan_to_num(rmse * mask)
        return np.mean(rmse)


def masked_mae_np(labels, preds, null_val=0, mode='dcrnn'):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        if mode == 'dcrnn':
            return np.mean(mae)
        else:
            return np.mean(mae, axis=(0, 1))


def mean_arctangent_absolute_percentage_error(labels, preds, null_val=0.0):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        maape = np.arctan(np.abs(labels-preds)/(labels+1e-8))
        maape = np.nan_to_num(maape * mask)
    return np.mean(maape)


def masked_mape_np(labels, preds, null_val=0):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)


def rmse_trainset(model, train_iter, batch_size, num, scaler, device=device):
    test = mx.nd.zeros(shape=(len(train_iter) * batch_size, num, 8), ctx=device)
    pred = mx.nd.zeros(shape=(len(train_iter) * batch_size, num, 8), ctx=device)
    count = 0

    for iter, (bg, label) in enumerate(train_iter):
        model.g = bg
        y = label
        edge_type = bg.edata['type']
        _ = model(bg, bg.ndata['h'], edge_type, None)
        test[count: count + len(label), :, :] = scaler.inverse_transform(y)
        pred[count: count + len(label), :] = scaler.inverse_transform(_).reshape(-1, num, 8)
        count += len(label)
    MSE = masked_mse_np(test.asnumpy().flatten(), pred.asnumpy().flatten())
    MAE = masked_mae_np(test.asnumpy().flatten(), pred.asnumpy().flatten())

    return np.sqrt(MSE), MSE / 2, MAE


def rmse_testset(model, _iter, batch_size, num_entra, scaler, mae=False, idx=None, plot=False, device=device):
    test = mx.nd.zeros(shape=(len(_iter) * batch_size, num_entra, 8), ctx=device)
    pred = mx.nd.zeros(shape=(len(_iter) * batch_size, num_entra, 8), ctx=device)
    count = 0
    for iter, (bg, label) in enumerate(_iter):
        y = label
        edge_type = bg.edata['type']
        _ = model(bg, bg.ndata['h'], edge_type, None)
        test[count: count + len(label), :] = scaler.inverse_transform(y)
        pred[count: count + len(label), :] = scaler.inverse_transform(_).reshape(-1, num_entra, 8)
        count += len(label)
    test = test.asnumpy()
    pred = pred.asnumpy()
    MSE = masked_mse_np(test.flatten(), pred.flatten())
    MAE = masked_mae_np(test.flatten(), pred.flatten())
    if mae:
        MAPE = masked_mape_np(test.flatten(), pred.flatten())
        return np.sqrt(MSE), MSE / 2, MAE, MAPE, test, pred
    else:
        return np.sqrt(MSE), MSE / 2, MSE / 2, MAE


class GraphTraffic(object):
    def __init__(self, num_graphs, num_nodes, matrix):
        super(GraphTraffic, self).__init__()
        self.num_graphs = num_graphs
        self.graphs = []
        self.labels = []
        self.matrix = matrix
        self.num_nodes = num_nodes
        self._generate()

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    @property
    def num_classes(self):
        return 1

    def _generate(self):
        self._gen_graph(self.num_graphs)

    def _gen_graph(self, n):
        for _ in range(n):
            num_v = self.num_nodes
            g = build_karate_club_graph(self.matrix)
            self.graphs.append(g)
            self.labels.append(0.0)


def collate(samples):
    # The input `samples` is a list of pairs (graph, label).
    graphs, label = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    ret = mx.nd.zeros(shape=(int(len(label)), 80, 8), ctx=device)
    for i in range(len(label)):
        ret[i] = label[i]
    return batched_graph, ret


def load_adjmatrix(filename):
    data = sio.loadmat(filename)
    tratim_matrix = np.array(data['adjacency_matrix'])
    # 构建网络图
    g = build_karate_club_graph(tratim_matrix)
#     nx_G = g.to_networkx()
#     pos = nx.kamada_kawai_layout(nx_G)
    print('number of nodes:', g.number_of_nodes())
    print('number of edges:', g.number_of_edges())
    return g, tratim_matrix


def build_karate_club_graph(matrix):
    g = dgl.DGLGraph().to(device)
    g.add_nodes(matrix.shape[0])
    rows, cols = np.where(matrix==1)
    g.add_edges(np.array(rows), np.array(cols))
    return g


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
        model.save_parameters('model.param')


def train_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)
    dgl.random.seed(seed)


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
