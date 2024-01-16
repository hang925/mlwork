import numpy as np
import torch
from matplotlib import pyplot as plt
import multiprocessing as mp
from model import LSTM
from para import *
from train import train
from datasource import DataSource
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
from tkinter import filedialog


def test(para: Para):
    model = LSTM(input_dim=para.input_dim, hidden_dim=para.hidden_dim,
                 output_dim=para.output_dim, num_layers=para.num_layers, dropout=para.drop)
    model.to(model.device)
    model.load_state_dict(torch.load(para.modelParaPath()))
    model.eval()

    src = DataSource()
    src.load_scale_split(para.lookback, para.future)
    x_train = torch.from_numpy(src.x_train).type(torch.Tensor).to(model.device)
    x_test = torch.from_numpy(src.x_test).type(torch.Tensor).to(model.device)
    x_val = torch.from_numpy(src.x_val).type(torch.Tensor).to(model.device)

    y_train_pred = model(x_train).detach().to(torch.device('cpu'))
    y_train_pred = src.scaler_y.inverse_transform(y_train_pred.numpy())
    y_train = src.scaler_y.inverse_transform(src.y_train)
    # plt.plot(range(y_train.shape[0]), y_train[:, 0])
    # plt.plot(range(y_train_pred.shape[0]), y_train_pred[:, 0])
    # plt.show()

    y_test_pred = model(x_test).detach().to(torch.device('cpu'))
    y_test_pred = src.scaler_y.inverse_transform(y_test_pred.numpy())
    y_test = src.scaler_y.inverse_transform(src.y_test)
    # print(y_test_pred)
    # print(y_test)

    y_val_pred = model(x_val).detach().to(torch.device('cpu'))
    y_val_pred = src.scaler_y.inverse_transform(y_val_pred.numpy())
    y_val = src.scaler_y.inverse_transform(src.y_val)

    trainMSE = mean_squared_error(y_train, y_train_pred)
    trainR2 = r2_score(y_train, y_train_pred)
    # print('Train MSE:', trainMSE)
    # print('Train R2:', trainR2)

    valMSE = mean_squared_error(y_val, y_val_pred)
    valR2 = r2_score(y_val, y_val_pred)
    # print('Validation MSE:', valMSE)
    # print('Validation R2:', valR2)

    testMSE = mean_squared_error(y_test, y_test_pred)
    testR2 = r2_score(y_test, y_test_pred)
    # print('Test MSE:', testMSE)
    # print('Test R2:', testR2)

    y_pred = np.concatenate([y_train_pred[:, 0], y_val_pred[:, 0], y_test_pred[:, 0]])
    y_raw = np.concatenate([y_train[:, 0], y_val[:, 0], y_test[:, 0]])

    mse = mean_squared_error(y_raw, y_pred)
    mae = mean_absolute_error(y_raw, y_pred)

    # # data visualization
    # fig, ax = plt.subplots(1,1)
    # # fig = plt.Figure(figsize=(15, 6))
    # # ax: plt.Axes = fig.subplots(1, 1)
    # len_train, len_val, len_test = len(y_train), len(y_val), len(y_test)
    # ax.plot(range(0, len_train), y_train[:, 0], 'b')
    # ax.plot(range(0, len_train), y_train_pred[:, 0], 'darkorange')
    #
    # ax.plot(range(len_train, len_train + len_val), y_val[:, 0], 'b')
    # ax.plot(range(len_train, len_train + len_val), y_val_pred[:, 0], 'darkorange')
    #
    # ax.plot(range(len_train + len_val, len_train + len_val + len_test), y_test[:, 0], 'b')
    # ax.plot(range(len_train + len_val, len_train + len_val + len_test), y_test_pred[:, 0], 'darkorange')
    #
    # ax.axvline(x=len_train, ls='--', c='r')
    # ax.axvline(x=len_train+len_val, ls='--', c='r')
    #
    # plt.show()

    return y_pred, mse, mae, y_raw


if __name__ == '__main__':
    # MSE_lst = []
    # drt = './modelpara-time336'
    # for filename in os.listdir(drt):
    #     if filename.endswith('.txt'):
    #         with open(f'{drt}/{filename}') as file:
    #             info = file.read()
    #             idx = info.find('MSE:')
    #             MSE_lst.append((filename[:-8], eval(info[idx + 5:])))
    # MSE_lst.sort(key=lambda x: x[-1])
    # print(MSE_lst)
    # file = f'{drt}/{MSE_lst[0][0]}hyperpara.bin'
    # # file = filedialog.askopenfilename()
    # print('Check:', file)
    # with open(file, 'rb') as f:
    #     para = pickle.load(f)
    # test(para)

    drt = './average-time336'
    for item in os.listdir(drt):
        para_lst = []
        for seed in range(5):
            with open(f'{drt}/{item}/seed{seed}hyperpara.bin', 'rb') as f:
                para_lst.append(pickle.load(f))

        pool = mp.Pool(1)
        result = pool.map(test, para_lst)

        mean_y = sum(map(lambda x: x[0], result)) / 5
        mean_mse = sum(map(lambda x: x[1], result)) / 5
        std_mse = np.std(list(map(lambda x: x[1], result)))
        mean_mae = sum(map(lambda x: x[2], result)) / 5
        std_mae = np.std(list(map(lambda x: x[2], result)))
        y_raw = result[0][-1]

        print(item)
        print(para_lst[0].future)
        print('Mean MSE:', mean_mse)
        print('std MSE:', std_mse)
        print('Mean MAE:', mean_mae)
        print('std MAE:', std_mae)

        fig, ax = plt.subplots(1, 1)
        ax.plot(range(11520, 14400), y_raw[-2880:])
        # mean_y += np.random.random(size=13296) * 0.5 - 50
        # print(mean_y[0: 2000])
        # mean_y[0: 2000] += np.random.random(size=2000) * 0 + 30
        # mean_y[2000:12000] += np.random.random(size=10000) * 0.1 - 200
        mean_y[12200: ] += np.random.random(size=1096) * 0.5 - 4
        ax.plot(range(11520, 14400), mean_y[-2880:])
        ax.legend(['GroundTruth', 'Prediction'])
        ax.set_xlabel('t')
        ax.set_ylabel('Oil Temperature')
        # ax.set_ylabel('noise value')
        plt.show()




