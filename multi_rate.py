# package
import os
import math
import time
import json
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import ParameterGrid
from scipy.interpolate import interp1d, splrep, splev
from sklearn.metrics import r2_score, mean_squared_error

# parameter parser
parser = argparse.ArgumentParser(description='multi-rate soft sensor')

# data parameters
parser.add_argument('-skiprows', type=int, default=23000)
parser.add_argument('-num_sample', type=int, default=12000)
parser.add_argument('-num_train', type=int, default=8000)
parser.add_argument('-num_val', type=int, default=2000)

# multi-rate parameters
parser.add_argument('-miss_value', type=int, default=0)
parser.add_argument('-x1', type=list, default=[0, 1, 2])
parser.add_argument('-x1_rate', type=int, default=1)
parser.add_argument('-x2',
                    type=list,
                    default=[11, 19, 22, 27, 28, 31, 32, 33, 34, 35])
parser.add_argument('-x2_rate', type=int, default=2)
parser.add_argument('-x3',
                    type=list,
                    default=[
                        3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16, 17, 18, 20,
                        23, 24, 25, 26, 29, 30
                    ])
parser.add_argument('-x3_rate', type=int, default=10)
parser.add_argument('-y', type=list, default=[9, 21])
parser.add_argument('-y_rate', type=int, default=20)

# model parameters
parser.add_argument('-model',
                    type=str,
                    default='MCW-RNN',
                    help='D-MLP|U-MLP|L-MLP|D-RNN|U-RNN|Z-RNN|MCW-RNN')
parser.add_argument('-kind', type=str, default='cubic')
parser.add_argument('-multiplier', type=int, default=15)
parser.add_argument('-dim_h', type=int, default=128)
parser.add_argument('-lr', type=float, default=0.001)
parser.add_argument('-weight_decay', type=float, default=0.1)
parser.add_argument('-step_size', type=int, default=50)
parser.add_argument('-gamma', type=float, default=0.5)
parser.add_argument('-epoch1', type=int, default=200)
parser.add_argument('-epoch2', type=int, default=400)
parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-report_freq', type=int, default=10)
parser.add_argument('-test_freq', type=int, default=20)

# other parameters
parser.add_argument('-path', type=str, default='multi_rate/')
parser.add_argument('-tag', type=str, default='tag/')
parser.add_argument('-figsize', type=tuple, default=(10, 10))
parser.add_argument('-dpi', type=int, default=150)
parser.add_argument('-seed', type=int, default=123)
parser.add_argument('-gpu', type=int, default=0)
parser.add_argument('-hpo', type=bool, default=True)

args = parser.parse_args()


def data_generation(data):
    data_new = data.copy()
    for i in range(args.num_sample):
        if (i + 1) % args.x1_rate != 0:
            data_new[i, args.x1] = args.miss_value
        if (i + 1) % args.x2_rate != 0:
            data_new[i, args.x2] = args.miss_value
        if (i + 1) % args.x3_rate != 0:
            data_new[i, args.x3] = args.miss_value
        if (i + 1) % args.y_rate != 0:
            data_new[i, args.y] = args.miss_value
    return data_new


def interpolation(data):
    n = data.shape[0]
    t = np.linspace(0, n - 1, n)
    for i in range(data.shape[1]):
        idx = data[:, i] != args.miss_value
        if sum(~idx) != 0:
            # tck = splrep(t[idx], data[idx, i], k=3, s=0)
            # data[~idx, i] = splev(t[~idx], tck, der=0)
            f = interp1d(t[idx],
                         data[idx, i],
                         kind=args.kind,
                         fill_value='extrapolate')
            data[~idx, i] = f(t[~idx])


def data_normalization(X_train, X_test):
    X_train_std = X_train.copy()
    X_test_std = X_test.copy()

    for i in range(X_train.shape[1]):
        index_train = X_train[:, i] != args.miss_value
        index_test = X_test[:, i] != args.miss_value
        min_value = min(X_train[index_train, i])
        max_value = max(X_train[index_train, i])
        X_train_std[index_train, i] = (X_train_std[index_train, i] -
                                       min_value) / (max_value - min_value)
        X_test_std[index_test, i] = (X_test_std[index_test, i] -
                                     min_value) / (max_value - min_value)

    return X_train_std, X_test_std


class MyDataset(Dataset):
    def __init__(self, X, y):
        super(Dataset, self).__init__()
        self.X = self.__transform__(X)
        self.y = self.__transform__(y)

    def __transform__(self, data):
        return torch.tensor(data, dtype=torch.float32).cuda(args.gpu)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.X.shape[0]


class MLP(nn.Module):
    def __init__(self, dim_x, multiplier, dim_h, dim_y):
        super(MLP, self).__init__()
        self.dim_x = dim_x
        self.multiplier = multiplier
        self.dim_h1 = dim_x * multiplier
        self.dim_h2 = dim_h
        self.dim_y = dim_y
        self.mlp = nn.ModuleList()
        for _ in range(dim_y):
            self.mlp.append(
                nn.Sequential(nn.Linear(self.dim_x, self.dim_h1), nn.ReLU(),
                              nn.Linear(self.dim_h1, self.dim_h2), nn.ReLU(),
                              nn.Linear(self.dim_h2, 1)))

    def forward(self, x):
        y = []
        for i in range(self.dim_y):
            y.append(self.mlp[i](x[:, -1]).squeeze())
        return torch.stack(y, dim=1)


class L_MLP(nn.Module):
    def __init__(self, dim_x, multiplier, dim_h, dim_y):
        super(L_MLP, self).__init__()
        self.dim_x = len(args.x1) * (args.y_rate // args.x1_rate) + len(
            args.x2) * (args.y_rate // args.x2_rate) + len(
                args.x3) * (args.y_rate // args.x3_rate)
        self.multiplier = multiplier
        self.dim_h1 = dim_x * multiplier
        self.dim_h2 = dim_h
        self.dim_y = dim_y
        self.mlp = nn.ModuleList()
        for _ in range(dim_y):
            self.mlp.append(
                nn.Sequential(nn.Linear(self.dim_x, self.dim_h1), nn.ReLU(),
                              nn.Linear(self.dim_h1, self.dim_h2), nn.ReLU(),
                              nn.Linear(self.dim_h2, 1)))

    def forward(self, x):
        y = []
        x_lift = torch.cat([
            x[:,
              list(range(args.x1_rate - 1, args.y_rate, args.x1_rate)
                   ), :len(args.x1)].reshape(x.shape[0], -1),
            x[:,
              list(range(args.x2_rate - 1, args.y_rate, args.x2_rate)),
              len(args.x1):len(args.x1) + len(args.x2)].reshape(
                  x.shape[0], -1),
            x[:,
              list(range(args.x3_rate - 1, args.y_rate, args.x3_rate)),
              -len(args.x3):].reshape(x.shape[0], -1)
        ],
                           dim=1)
        for i in range(self.dim_y):
            y.append(self.mlp[i](x_lift).squeeze())
        return torch.stack(y, dim=1)


class RNN(nn.Module):
    def __init__(self, dim_x, multiplier, dim_h, dim_y):
        super(RNN, self).__init__()
        self.dim_x = dim_x
        self.multiplier = multiplier
        self.dim_h1 = dim_x * multiplier
        self.dim_h2 = dim_h
        self.dim_y = dim_y
        self.rnn = nn.ModuleList()
        self.mlp = nn.ModuleList()
        for _ in range(dim_y):
            self.rnn.append(nn.RNN(dim_x, self.dim_h1, batch_first=True))
            self.mlp.append(
                nn.Sequential(nn.Linear(self.dim_h1, self.dim_h2), nn.ReLU(),
                              nn.Linear(self.dim_h2, 1)))

    def forward(self, x):
        y = []
        for i in range(self.dim_y):
            _, h = self.rnn[i](x)
            y.append(self.mlp[i](h).squeeze())
        return torch.stack(y, dim=1)


class MCW_RNN(nn.Module):
    def __init__(self, dim_x, multiplier, dim_h, dim_y):
        super(MCW_RNN, self).__init__()
        self.dim_x = dim_x
        self.multiplier = multiplier
        self.dim_h1 = dim_x * multiplier
        self.dim_h2 = dim_h
        self.dim_y = dim_y
        self.w_x = nn.ParameterList()
        self.w_h = nn.ParameterList()
        self.b = nn.ParameterList()
        self.act = nn.Tanh()
        self.mlp = nn.ModuleList()
        for _ in range(dim_y):
            self.w_x.append(nn.Parameter(torch.FloatTensor(dim_x,
                                                           self.dim_h1)))
            self.w_h.append(
                nn.Parameter(torch.FloatTensor(self.dim_h1, self.dim_h1)))
            self.b.append(nn.Parameter(torch.FloatTensor(self.dim_h1)))
            self.mlp.append(
                nn.Sequential(nn.Linear(self.dim_h1, self.dim_h2), nn.ReLU(),
                              nn.Linear(self.dim_h2, 1)))
        self.reset()

    def forward(self, x, mode='slowest'):
        '''
        mode
        ----------
        slowest: predict quality variable at the slowest sampling rate.

        fastest: predict quality variable at the fastest sampling rate.

        best: predict quality variable at the fastest sampling rate, where
        each sample is predicted with same sequence length.
        '''
        y_list = []
        for i in range(self.dim_y):
            if mode == 'best':
                y = []
                for j in range(x.shape[0]):
                    h1 = torch.zeros(len(args.x1) * self.multiplier).cuda(
                        args.gpu)
                    h2 = torch.zeros(len(args.x2) * self.multiplier).cuda(
                        args.gpu)
                    h3 = torch.zeros(len(args.x3) * self.multiplier).cuda(
                        args.gpu)
                    for k in range(x.shape[1]):
                        x_t = x[j, k]
                        h = torch.cat((h1, h2, h3), dim=0)
                        if sum(x_t != args.miss_value) == len(args.x1):
                            h1 = self.act(
                                torch.matmul(
                                    x_t[:len(args.x1)], self.w_x[i]
                                    [:len(args.x1), :h1.shape[0]]) +
                                torch.matmul(h, self.w_h[i][:, :h1.shape[0]]) +
                                self.b[i][:h1.shape[0]])
                        elif sum(x_t != args.miss_value) == len(args.x1) + len(
                                args.x2):
                            h1 = self.act(
                                torch.matmul(
                                    x_t[:len(args.x1) + len(args.x2)],
                                    self.w_x[i][:len(args.x1) +
                                                len(args.x2), :h1.shape[0]]) +
                                torch.matmul(h, self.w_h[i][:, :h1.shape[0]]) +
                                self.b[i][:h1.shape[0]])
                            h2 = self.act(
                                torch.matmul(
                                    x_t[:len(args.x1) +
                                        len(args.x2)], self.w_x[i]
                                    [:len(args.x1) + len(args.x2),
                                     h1.shape[0]:h1.shape[0] + h2.shape[0]]) +
                                torch.matmul(
                                    h, self.w_h[i][:, h1.shape[0]:h1.shape[0] +
                                                   h2.shape[0]]) +
                                self.b[i][h1.shape[0]:h1.shape[0] +
                                          h2.shape[0]])
                        else:
                            h1 = self.act(
                                torch.matmul(x_t, self.w_x[i]
                                             [:, :h1.shape[0]]) +
                                torch.matmul(h, self.w_h[i][:, :h1.shape[0]]) +
                                self.b[i][:h1.shape[0]])
                            h2 = self.act(
                                torch.matmul(
                                    x_t, self.w_x[i][:,
                                                     h1.shape[0]:h1.shape[0] +
                                                     h2.shape[0]]) +
                                torch.matmul(
                                    h, self.w_h[i][:, h1.shape[0]:h1.shape[0] +
                                                   h2.shape[0]]) +
                                self.b[i][h1.shape[0]:h1.shape[0] +
                                          h2.shape[0]])
                            h3 = self.act(
                                torch.matmul(x_t, self.w_x[i][:,
                                                              -h3.shape[0]:]) +
                                torch.matmul(h, self.w_h[i][:,
                                                            -h3.shape[0]:]) +
                                self.b[i][-h3.shape[0]:])
                    y.append(self.mlp[i](torch.cat((h1, h2, h3),
                                                   dim=0)).squeeze())
                y_list.append(torch.stack(y, dim=0).reshape(-1))
            else:
                y = []
                h1 = torch.zeros(x.shape[0],
                                 len(args.x1) * self.multiplier).cuda(args.gpu)
                h2 = torch.zeros(x.shape[0],
                                 len(args.x2) * self.multiplier).cuda(args.gpu)
                h3 = torch.zeros(x.shape[0],
                                 len(args.x3) * self.multiplier).cuda(args.gpu)
                for j in range(x.shape[1]):
                    h = torch.cat((h1, h2, h3), dim=1)
                    if (j + 1) % args.x2_rate == 0:
                        if (j + 1) % args.x3_rate == 0:
                            h1 = self.act(
                                torch.matmul(x[:, j], self.w_x[i]
                                             [:, :h1.shape[1]]) +
                                torch.matmul(h, self.w_h[i][:, :h1.shape[1]]) +
                                self.b[i][:h1.shape[1]])
                            h2 = self.act(
                                torch.matmul(
                                    x[:,
                                      j], self.w_x[i][:,
                                                      h1.shape[1]:h1.shape[1] +
                                                      h2.shape[1]]) +
                                torch.matmul(
                                    h, self.w_h[i][:, h1.shape[1]:h1.shape[1] +
                                                   h2.shape[1]]) +
                                self.b[i][h1.shape[1]:h1.shape[1] +
                                          h2.shape[1]])
                            h3 = self.act(
                                torch.matmul(x[:, j], self.w_x[i]
                                             [:, -h3.shape[1]:]) +
                                torch.matmul(h, self.w_h[i][:,
                                                            -h3.shape[1]:]) +
                                self.b[i][-h3.shape[1]:])
                        else:
                            h1 = self.act(
                                torch.matmul(
                                    x[:, j, :len(args.x1) + len(args.x2)],
                                    self.w_x[i][:len(args.x1) +
                                                len(args.x2), :h1.shape[1]]) +
                                torch.matmul(h, self.w_h[i][:, :h1.shape[1]]) +
                                self.b[i][:h1.shape[1]])
                            h2 = self.act(
                                torch.matmul(
                                    x[:, j, :len(args.x1) +
                                      len(args.x2)], self.w_x[i]
                                    [:len(args.x1) + len(args.x2),
                                     h1.shape[1]:h1.shape[1] + h2.shape[1]]) +
                                torch.matmul(
                                    h, self.w_h[i][:, h1.shape[1]:h1.shape[1] +
                                                   h2.shape[1]]) +
                                self.b[i][h1.shape[1]:h1.shape[1] +
                                          h2.shape[1]])
                    else:
                        h1 = self.act(
                            torch.matmul(
                                x[:, j, :len(args.x1)], self.w_x[i]
                                [:len(args.x1), :h1.shape[1]]) +
                            torch.matmul(h, self.w_h[i][:, :h1.shape[1]]) +
                            self.b[i][:h1.shape[1]])
                    if mode == 'fastest':
                        y.append(self.mlp[i](torch.cat((h1, h2, h3),
                                                       dim=1)).squeeze())
                if mode == 'fastest':
                    y_list.append(torch.stack(y, dim=1).reshape(-1))
                elif mode == 'slowest':
                    y_list.append(self.mlp[i](torch.cat((h1, h2, h3),
                                                        dim=1)).squeeze())
                else:
                    raise Exception('Wrong mode selection!')
        return torch.stack(y_list, dim=1)

    def reset(self):
        std = 1. / math.sqrt(self.dim_h1)
        for i in range(self.dim_y):
            self.w_x[i].data.uniform_(-std, std)
            self.w_h[i].data.uniform_(-std, std)
            self.b[i].data.uniform_(-std, std)


def train(X_train,
          y_train,
          X_test,
          y_test,
          scaler,
          mode,
          X_test_best=None,
          y_test_best=None):
    y_train_raw = scaler.inverse_transform(y_train)

    # mode selection
    if mode == 'val':
        y_test = scaler.inverse_transform(y_test)
        n_epoch = args.epoch1
        test_freq = n_epoch
    elif mode == 'test':
        n_epoch = args.epoch2
        test_freq = args.test_freq
    else:
        raise Exception('Wrong mode selection!')

    # model initialization
    if args.model == 'MCW-RNN':
        net = MCW_RNN(X_train.shape[-1], args.multiplier, args.dim_h,
                      y_train.shape[-1]).cuda(args.gpu)
    elif args.model in ['D-RNN', 'U-RNN', 'Z-RNN']:
        net = RNN(X_train.shape[-1], args.multiplier, args.dim_h,
                  y_train.shape[-1]).cuda(args.gpu)
    elif args.model == 'L-MLP':
        net = L_MLP(X_train.shape[-1], args.multiplier, args.dim_h,
                    y_train.shape[-1]).cuda(args.gpu)
    elif args.model in ['D-MLP', 'U-MLP']:
        net = MLP(X_train.shape[-1], args.multiplier, args.dim_h,
                  y_train.shape[-1]).cuda(args.gpu)
    else:
        raise Exception('Wrong model selection!')
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(net.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer,
                                    step_size=args.step_size,
                                    gamma=args.gamma)

    # data initialization
    dataset_train = MyDataset(X_train, y_train)
    dataset_test = MyDataset(X_test, y_test)
    dataloader = DataLoader(dataset_train,
                            batch_size=args.batch_size,
                            shuffle=True)

    # training
    loss_hist = np.zeros(n_epoch)
    acc_train = np.zeros((n_epoch // test_freq, len(args.y)))
    acc_test = np.zeros((n_epoch // test_freq, len(args.y)))
    for epoch in range(n_epoch):
        start = time.time()
        net.train()
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            output = net(batch_X)
            loss = criterion(batch_y, output)
            loss.backward()
            loss_hist[epoch] += loss.item()
            optimizer.step()
        scheduler.step()
        end = time.time()

        if (epoch + 1) % args.report_freq == 0:
            print('Epoch: {:03d}, Loss: {:.3f}, Time: {:.2f}s'.format(
                epoch + 1, loss_hist[epoch], end - start))

        if (epoch + 1) % test_freq == 0:
            net.eval()
            with torch.no_grad():
                y_fit = scaler.inverse_transform(
                    net(dataset_train.X).cpu().numpy())
                y_pred = scaler.inverse_transform(
                    net(dataset_test.X).cpu().numpy())
            r2_train = [
                round(100 * _, 2)
                for _ in r2_score(y_train_raw, y_fit, multioutput='raw_values')
            ]
            r2_test = [
                round(100 * _, 2)
                for _ in r2_score(y_test, y_pred, multioutput='raw_values')
            ]
            rmse_train = [
                round(math.sqrt(_), 3) for _ in mean_squared_error(
                    y_train_raw, y_fit, multioutput='raw_values')
            ]
            rmse_test = [
                round(math.sqrt(_), 3) for _ in mean_squared_error(
                    y_test, y_pred, multioutput='raw_values')
            ]
            print('-' * 50)
            print('Performance on train set: R2: {}, RMSE: {}'.format(
                r2_train, rmse_train))
            print('Performance on test set: R2: {}, RMSE: {}'.format(
                r2_test, rmse_test))
            print('-' * 50)
            acc_train[(epoch + 1) // test_freq - 1, :] = r2_train
            acc_test[(epoch + 1) // test_freq - 1, :] = r2_test
    print('Training is finished!')

    # plot
    if mode == 'test':
        plot_loss_acc(loss_hist, acc_train, acc_test)
        plot_prediction_curve(y_train_raw, y_fit, r2_train, rmse_train, y_test,
                              y_pred, r2_test, rmse_test)

        if args.model == 'MCW-RNN':

            # performance on different time steps
            net.eval()
            with torch.no_grad():
                y_pred = scaler.inverse_transform(
                    net(dataset_test.X, 'fastest').cpu().numpy())
            r2 = np.zeros((args.y_rate, len(args.y)))
            rmse = np.zeros((args.y_rate, len(args.y)))
            for i in range(args.y_rate):
                idx = range(i, y_test_best.shape[0], args.y_rate)
                r2[i] = 100 * r2_score(
                    y_test_best[idx], y_pred[idx], multioutput='raw_values')
                rmse[i] = np.sqrt(
                    mean_squared_error(y_test_best[idx],
                                       y_pred[idx],
                                       multioutput='raw_values'))

            # plot R2
            plt.figure(figsize=args.figsize, dpi=args.dpi)
            for i in range(len(args.y)):
                plt.subplot(len(args.y), 1, i + 1)
                plt.bar(range(1, args.y_rate + 1),
                        r2[:, i],
                        label='QV_{}'.format(i + 1))
                plt.xlabel('Time step')
                plt.ylabel('R2')
                plt.grid()
                plt.legend()
                plt.title('R2 on different time steps (QV_{})'.format(i + 1))
            plt.savefig(args.path + args.tag + 'R2.png')
            plt.close()

            # plot RMSE
            plt.figure(figsize=args.figsize, dpi=args.dpi)
            for i in range(len(args.y)):
                plt.subplot(len(args.y), 1, i + 1)
                plt.bar(range(1, args.y_rate + 1),
                        rmse[:, i],
                        label='QV_{}'.format(i + 1))
                plt.xlabel('Time step')
                plt.ylabel('RMSE')
                plt.grid()
                plt.legend()
                plt.title('RMSE on different time step (QV_{})'.format(i + 1))
            plt.savefig(args.path + args.tag + 'RMSE.png')
            plt.close()

            # best performance
            dataset_best = MyDataset(X_test_best, y_test_best)
            with torch.no_grad():
                y_pred = scaler.inverse_transform(
                    net(dataset_best.X, 'best').cpu().numpy())
            r2 = [
                round(100 * _, 2)
                for _ in r2_score(y_test_best[args.y_rate - 1:],
                                  y_pred,
                                  multioutput='raw_values')
            ]
            rmse = [
                round(math.sqrt(_), 3)
                for _ in mean_squared_error(y_test_best[args.y_rate - 1:],
                                            y_pred,
                                            multioutput='raw_values')
            ]

            # plot
            plt.figure(figsize=args.figsize, dpi=args.dpi)
            for i in range(len(args.y)):
                plt.subplot(len(args.y), 1, i + 1)
                plt.plot(y_test_best[args.y_rate - 1:, i],
                         label='Ground Truth')
                plt.plot(y_pred[:, i], label='Prediction')
                plt.xlabel('Sample')
                plt.ylabel('Value')
                plt.grid()
                plt.legend()
                plt.title(
                    'Prediction curve of QV{} (R2= {:.2f}%, RMSE= {:.3f})'.
                    format(i + 1, r2[i], rmse[i]))
            plt.savefig(args.path + args.tag + 'Best.png')
            plt.close()

    return acc_train, acc_test


def plot_loss_acc(loss_hist, acc_train, acc_test):
    plt.figure(figsize=args.figsize, dpi=args.dpi)
    plt.subplot(211)
    plt.plot(list(range(1, loss_hist.shape[0] + 1)), loss_hist, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.grid()
    plt.legend()
    plt.title('Loss curve during training')
    plt.subplot(212)
    for i in range(len(args.y)):
        plt.plot([
            args.test_freq * (_ + 1)
            for _ in range(loss_hist.shape[0] // args.test_freq)
        ],
                 acc_train[:, i],
                 label='QV{}(train)'.format(i + 1))
        plt.plot([
            args.test_freq * (_ + 1)
            for _ in range(loss_hist.shape[0] // args.test_freq)
        ],
                 acc_test[:, i],
                 label='QV{}(test)'.format(i + 1))
    plt.xlabel('Epoch')
    plt.ylabel('R2 Value')
    plt.grid()
    plt.legend()
    plt.title('Accuracy curve during training')
    plt.savefig(args.path + args.tag + 'Loss & Acc.png')
    plt.close()


def plot_prediction_curve(y_train, y_fit, r2_train, rmse_train, y_test, y_pred,
                          r2_test, rmse_test):
    for i in range(len(args.y)):
        plt.figure(figsize=args.figsize, dpi=args.dpi)
        plt.subplot(211)
        plt.plot(y_train[:, i], label='Ground Truth')
        plt.plot(y_fit[:, i], label='Prediciton')
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.grid()
        plt.legend()
        plt.title(
            'Prediction curve on train set (R2: {:.2f}%, RMSE: {:.3f})'.format(
                r2_train[i], rmse_train[i]))
        plt.subplot(212)
        plt.plot(y_test[:, i], label='Ground Truth')
        plt.plot(y_pred[:, i], label='Prediciton')
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.grid()
        plt.legend()
        plt.title(
            'Prediction curve on test set (R2: {:.2f}%, RMSE: {:.3f})'.format(
                r2_test[i], rmse_test[i]))
        plt.savefig(args.path + args.tag + 'QV_{}.png'.format(i + 1))
        plt.close()


# main
def main():

    # seed fixed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # create folder
    if not os.path.exists(args.path + args.tag):
        os.mkdir(args.path + args.tag)

    # import data
    data = pd.read_excel(args.path + 'coal_mill.xlsx',
                         header=1,
                         index_col=0,
                         skiprows=[2])
    # columns = data.columns
    # for i in range(data.shape[1]):
    #     plt.figure(figsize=args.figsize, dpi=args.dpi)
    #     plt.plot(data.iloc[:, i])
    #     plt.xlabel('Sample')
    #     plt.ylabel('Value')
    #     plt.grid()
    #     plt.savefig(args.path + args.tag + columns[i] + '.png')
    #     plt.close()
    data = data.values[args.skiprows:args.skiprows + args.num_sample]

    # data generation
    data_new = data_generation(data)

    # data split
    X, y = data_new[:, args.x1 + args.x2 + args.x3], data_new[:, args.y]
    X_train, y_train = X[:args.num_train], y[:args.num_train]
    X_test, y_test = X[args.num_train:], y[args.num_train:]

    # U-MLP
    if args.model == 'U-MLP':
        # interpolation
        interpolation(X_train)
        interpolation(y_train)
        idx = y_test[:, 0] != args.miss_value
        X_test = X_test[idx]
        y_test = y_test[idx]

        # data normalization
        scaler = MinMaxScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        scaler = MinMaxScaler().fit(y_train)
        y_train = scaler.transform(y_train)

        # data rearrangement
        X_train_3d = X_train[:, np.newaxis, :]
        X_test_3d = X_test[:, np.newaxis, :]

    # U-RNN
    elif args.model == 'U-RNN':
        # interpolation
        interpolation(X_train)
        interpolation(y_train)
        interpolation(X_test)
        y_test = y_test[y_test[:, 0] != args.miss_value]

        # data normalization
        scaler = MinMaxScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        scaler = MinMaxScaler().fit(y_train)
        y_train = scaler.transform(y_train)

        # data rearrangement
        X_train_3d = []
        X_test_3d = []
        for i in range(X_train.shape[0] - args.y_rate + 1):
            X_train_3d.append(X_train[i:i + args.y_rate])
        for i in range(X_test.shape[0] // args.y_rate):
            X_test_3d.append(X_test[i * args.y_rate:(i + 1) * args.y_rate])
        X_train_3d = np.stack(X_train_3d)
        X_test_3d = np.stack(X_test_3d)
        y_train = y_train[args.y_rate - 1:]

    # D-RNN
    elif args.model == 'D-RNN':
        # down sampling
        idx = y_train[:, 0] != args.miss_value
        X_train = X_train[idx]
        y_train = y_train[idx]
        idx = y_test[:, 0] != args.miss_value
        X_test = X_test[idx]
        y_test = y_test[idx]

        # data normalization
        scaler = MinMaxScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        scaler = MinMaxScaler().fit(y_train)
        y_train = scaler.transform(y_train)

        # data rearrangement
        X_train_3d = []
        X_test_3d = []
        for i in range(X_train.shape[0] - args.y_rate + 1):
            X_train_3d.append(X_train[i:i + args.y_rate])
        for i in range(X_test.shape[0] - args.y_rate + 1):
            X_test_3d.append(X_test[i:i + args.y_rate])
        X_train_3d = np.stack(X_train_3d)
        X_test_3d = np.stack(X_test_3d)
        y_train = y_train[args.y_rate - 1:]
        y_test = y_test[args.y_rate - 1:]

    # others
    else:
        # data normalization
        X_train, X_test = data_normalization(X_train, X_test)
        y_train = y_train[y_train[:, 0] != args.miss_value]
        y_test = y_test[y_test[:, 0] != args.miss_value]
        scaler = MinMaxScaler().fit(y_train)
        y_train = scaler.transform(y_train)

        # data rearrangement
        X_train_3d = []
        X_test_3d = []
        X_test_3d_best = []
        for i in range(X_train.shape[0] // args.y_rate):
            X_train_3d.append(X_train[i * args.y_rate:(i + 1) * args.y_rate])
        for i in range(X_test.shape[0] // args.y_rate):
            X_test_3d.append(X_test[i * args.y_rate:(i + 1) * args.y_rate])
        for i in range(X_test.shape[0] - args.y_rate + 1):
            X_test_3d_best.append(X_test[i:i + args.y_rate])
        X_train_3d = np.stack(X_train_3d)
        X_test_3d = np.stack(X_test_3d)
        X_test_3d_best = np.stack(X_test_3d_best)
        y_test_best = data[args.num_train:, args.y]

    # model selection
    if args.hpo:
        print('=' * 50)
        print('Model selection')
        print('=' * 50)
        param_grid = {
            'multiplier': [15, 30, 60],
            'dim_h': [256, 128, 64],
            # 'lr': [0.001, 0.0001],
            'weight_decay': [0.1, 0.05, 0.01, 0.005],
            'batch_size': [64, 128]
        }
        record = {}
        r2_best = float('-inf')
        if args.model in ['U-MLP', 'U-RNN']:
            num_val = args.num_val
        else:
            num_val = args.num_val // args.y_rate
        for i, param in enumerate(ParameterGrid(param_grid)):
            print('Model-{}'.format(i + 1))
            print(param)
            for key, value in param.items():
                exec('args.{}={}'.format(key, value))
            r2_train, r2_val = train(X_train_3d[:-num_val], y_train[:-num_val],
                                     X_train_3d[-num_val:], y_train[-num_val:],
                                     scaler, 'val')
            torch.cuda.empty_cache()
            print('*' * 50)
            record.update({str(param): [list(r2_train[-1]), list(r2_val[-1])]})
            if sum(r2_val[-1]) > r2_best:
                r2_best = sum(r2_val[-1])
                param_best = param
        record.update({'best': param_best})
        for key, value in param_best.items():
            exec('args.{}={}'.format(key, value))
        with open(args.path + args.tag + 'record.json', 'w') as f:
            json.dump(record, f)
        print(param_best)

    # retraining
    print('=' * 50)
    print('Model Retraining')
    print('=' * 50)
    if args.model == 'MCW-RNN':
        train(X_train_3d, y_train, X_test_3d, y_test, scaler, 'test',
              X_test_3d_best, y_test_best)
    else:
        train(X_train_3d, y_train, X_test_3d, y_test, scaler, 'test')

    # end
    pass


if __name__ == '__main__':
    main()