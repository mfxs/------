# package
import math
import time
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
from sklearn.metrics import r2_score, mean_squared_error

# parameter parser
parser = argparse.ArgumentParser(description='multi-rate soft sensor')

# data parameters
parser.add_argument('-nrows', type=int, default=12000)
parser.add_argument('-num_train', type=int, default=8000)
parser.add_argument('-skiprows', type=int, default=23000)

# multi-rate parameters
parser.add_argument('-miss_value', type=int, default=-1)
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
parser.add_argument('-dim_h1', type=int, default=30)
parser.add_argument('-dim_h2', type=int, default=256)
parser.add_argument('-lr', type=float, default=0.001)
parser.add_argument('-weight_decay', type=float, default=0.1)
parser.add_argument('-step_size', type=int, default=50)
parser.add_argument('-gamma', type=float, default=0.5)
parser.add_argument('-epoch', type=int, default=400)
parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-test_freq', type=int, default=20)

# other parameters
parser.add_argument('-path', type=str, default='multi_rate/')
parser.add_argument('-seed', type=int, default=123)
parser.add_argument('-gpu', type=int, default=2)

args = parser.parse_args()


def data_generation(data):
    for i in range(args.nrows):
        if (i + 1) % args.x1_rate != 0:
            data[i, args.x1] = args.miss_value
        if (i + 1) % args.x2_rate != 0:
            data[i, args.x2] = args.miss_value
        if (i + 1) % args.x3_rate != 0:
            data[i, args.x3] = args.miss_value
        if (i + 1) % args.y_rate != 0:
            data[i, args.y] = args.miss_value


def data_normalization(X_train, X_test):
    min_list = []
    max_list = []
    X_train_std = X_train.copy()
    X_test_std = X_test.copy()

    for i in range(X_train.shape[1]):
        min_list.append(min(X_train[X_train[:, i] != args.miss_value, i]))
        max_list.append(max(X_train[X_train[:, i] != args.miss_value, i]))
        X_train_std[X_train_std[:, i] != args.miss_value,
                    i] = (X_train_std[X_train_std[:, i] != args.miss_value, i]
                          - min_list[-1]) / (max_list[-1] - min_list[-1])
        X_test_std[X_test_std[:, i] != args.miss_value,
                   i] = (X_test_std[X_test_std[:, i] != args.miss_value, i] -
                         min_list[-1]) / (max_list[-1] - min_list[-1])

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


class CWRNN(nn.Module):
    def __init__(self, dim_x, dim_h1, dim_h2, dim_y):
        super(CWRNN, self).__init__()
        self.dim_x = dim_x
        self.dim_h1 = dim_h1
        self.dim_h2 = dim_h2
        self.dim_y = dim_y
        self.w_x = nn.Parameter(torch.FloatTensor(dim_x, dim_x * dim_h1))
        self.w_h = nn.Parameter(
            torch.FloatTensor(dim_x * dim_h1, dim_x * dim_h1))
        self.b = nn.Parameter(torch.FloatTensor(dim_x * dim_h1))
        self.act = nn.Tanh()
        self.mlp = nn.Sequential(nn.Linear(dim_x * dim_h1, dim_h2), nn.ReLU(),
                                 nn.Linear(dim_h2, dim_y))
        self.reset()

    def forward(self, x):
        h1 = torch.zeros(x.shape[0], len(args.x1) * self.dim_h1).cuda(args.gpu)
        h2 = torch.zeros(x.shape[0], len(args.x2) * self.dim_h1).cuda(args.gpu)
        h3 = torch.zeros(x.shape[0], len(args.x3) * self.dim_h1).cuda(args.gpu)
        for i in range(x.shape[1]):
            h = torch.cat((h1, h2, h3), dim=1)
            if (i + 1) % args.x2_rate == 0:
                if (i + 1) % args.x3_rate == 0:
                    h1 = self.act(
                        torch.matmul(x[:, i], self.w_x[:, :len(args.x1) *
                                                       self.dim_h1]) +
                        torch.matmul(h, self.w_h[:, :len(args.x1) *
                                                 self.dim_h1]))
                    h2 = self.act(
                        torch.matmul(
                            x[:, i], self.w_x[:,
                                              len(args.x1) * self.dim_h1:
                                              (len(args.x1) + len(args.x2)) *
                                              self.dim_h1]) +
                        torch.matmul(
                            h, self.w_h[:,
                                        len(args.x1) * self.dim_h1:
                                        (len(args.x1) + len(args.x2)) *
                                        self.dim_h1]))
                    h3 = self.act(
                        torch.matmul(x[:, i], self.w_x[:, -len(args.x3) *
                                                       self.dim_h1:]) +
                        torch.matmul(h, self.w_h[:, -len(args.x3) *
                                                 self.dim_h1:]))
                else:
                    h1 = self.act(
                        torch.matmul(
                            x[:, i, :len(args.x1) + len(args.x2)], self.
                            w_x[:len(args.x1) + len(args.x2), :len(args.x1) *
                                self.dim_h1]) +
                        torch.matmul(h, self.w_h[:, :len(args.x1) *
                                                 self.dim_h1]))
                    h2 = self.act(
                        torch.matmul(
                            x[:, i, :len(args.x1) + len(args.x2)], self.
                            w_x[:len(args.x1) + len(args.x2),
                                len(args.x1) * self.dim_h1:
                                (len(args.x1) + len(args.x2)) * self.dim_h1]) +
                        torch.matmul(
                            h, self.w_h[:,
                                        len(args.x1) * self.dim_h1:
                                        (len(args.x1) + len(args.x2)) *
                                        self.dim_h1]))
            else:
                h1 = self.act(
                    torch.matmul(
                        x[:, i, :len(args.x1)],
                        self.w_x[:len(args.x1), :len(args.x1) * self.dim_h1]) +
                    torch.matmul(h, self.w_h[:, :len(args.x1) * self.dim_h1]))

        y = self.mlp(torch.cat((h1, h2, h3), dim=1))
        return y

    def reset(self):
        std = 1. / math.sqrt(self.dim_h1)
        self.w_x.data.uniform_(-std, std)
        self.w_h.data.uniform_(-std, std)
        self.b.data.uniform_(-std, std)


# main
def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # import data
    data = pd.read_excel(args.path + 'coal_mill.xlsx',
                         header=1,
                         index_col=0,
                         skiprows=[2])
    # columns = data.columns
    # for i in range(data.shape[1]):
    #     plt.figure()
    #     plt.plot(data.iloc[:, i])
    #     plt.xlabel('Sample')
    #     plt.ylabel('Value')
    #     plt.grid()
    #     plt.savefig(args.path + columns[i] + '.png')
    #     plt.close()
    data = data.values[args.skiprows:args.skiprows + args.nrows]

    # data generation
    data_generation(data)

    # data split
    X, y = data[:, args.x1 + args.x2 + args.x3], data[:, args.y]
    X_train, y_train = X[:args.num_train], y[:args.num_train]
    X_test, y_test = X[args.num_train:], y[args.num_train:]

    # data normalization
    X_train_std, X_test_std = data_normalization(X_train, X_test)
    y_train = y_train[y_train[:, 0] != args.miss_value]
    y_test = y_test[y_test[:, 0] != args.miss_value]
    scaler = MinMaxScaler().fit(y_train)
    y_train_std = scaler.transform(y_train)

    # data rearrangement
    X_train_3d = []
    X_test_3d = []
    for i in range(X_train_std.shape[0] // args.y_rate):
        X_train_3d.append(X_train_std[i * args.y_rate:(i + 1) * args.y_rate])
    for i in range(X_test_std.shape[0] // args.y_rate):
        X_test_3d.append(X_test_std[i * args.y_rate:(i + 1) * args.y_rate])
    X_train_3d = np.stack(X_train_3d)
    X_test_3d = np.stack(X_test_3d)

    # model initialization
    net = CWRNN(X_train.shape[-1], args.dim_h1, args.dim_h2,
                y_train.shape[-1]).cuda(args.gpu)
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(net.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer,
                                    step_size=args.step_size,
                                    gamma=args.gamma)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.tmax)

    # data initialization
    dataset_train = MyDataset(X_train_3d, y_train_std)
    dataset_test = MyDataset(X_test_3d, y_test)
    dataloader = DataLoader(dataset_train,
                            batch_size=args.batch_size,
                            shuffle=True)

    # training
    loss_hist = np.zeros(args.epoch)
    acc_train = np.zeros((args.epoch // args.test_freq, len(args.y)))
    acc_test = np.zeros((args.epoch // args.test_freq, len(args.y)))
    for epoch in range(args.epoch):
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

        print('Epoch: {:03d}, Loss: {:.3f}, Time: {:.2f}s'.format(
            epoch + 1, loss_hist[epoch], end - start))

        if (epoch + 1) % args.test_freq == 0:
            net.eval()
            with torch.no_grad():
                y_fit = scaler.inverse_transform(
                    net(dataset_train.X).cpu().numpy())
                y_pred = scaler.inverse_transform(
                    net(dataset_test.X).cpu().numpy())
            r2_train = [
                100 * _
                for _ in r2_score(y_train, y_fit, multioutput='raw_values')
            ]
            r2_test = [
                100 * _
                for _ in r2_score(y_test, y_pred, multioutput='raw_values')
            ]
            rmse_train = [
                math.sqrt(_) for _ in mean_squared_error(
                    y_train, y_fit, multioutput='raw_values')
            ]
            rmse_test = [
                math.sqrt(_) for _ in mean_squared_error(
                    y_test, y_pred, multioutput='raw_values')
            ]
            print('=' * 50)
            print('Performance on train set: R2: {}, RMSE: {}'.format(
                r2_train, rmse_train))
            print('Performance on test set: R2: {}, RMSE: {}'.format(
                r2_test, rmse_test))
            print('=' * 50)
            acc_train[(epoch + 1) // args.test_freq - 1, :] = r2_train
            acc_test[(epoch + 1) // args.test_freq - 1, :] = r2_test
    print('Training is finished!')

    # plot loss & acc
    plt.figure()
    plt.subplot(211)
    plt.plot(list(range(1, args.epoch + 1)), loss_hist)
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.grid()
    plt.title('Loss curve during training')
    plt.subplot(212)
    for i in range(len(args.y)):
        plt.plot([20 * (_ + 1) for _ in range(args.epoch // args.test_freq)],
                 acc_train[:, i],
                 label='QV{}(train)'.format(i + 1))
        plt.plot([20 * (_ + 1) for _ in range(args.epoch // args.test_freq)],
                 acc_test[:, i],
                 label='QV{}(test)'.format(i + 1))
    plt.xlabel('Epoch')
    plt.ylabel('R2 Value')
    plt.grid()
    plt.legend()
    plt.title('Accuracy curve during training')
    plt.savefig(args.path + 'Loss & Acc.png')
    plt.close()

    # plot prediction curve
    for i in range(len(args.y)):
        plt.figure()
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
        plt.savefig(args.path + 'QV_{}.png'.format(i + 1))
        plt.close()

    # end
    pass


if __name__ == '__main__':
    main()