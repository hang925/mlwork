import torch
from matplotlib import pyplot as plt

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
    x_train, y_train, x_val, y_val, x_test, y_test, scaler = src.fetch(para.mode)

    x_train = torch.from_numpy(x_train).type(torch.Tensor).to(model.device)
    x_test = torch.from_numpy(x_test).type(torch.Tensor).to(model.device)
    x_val = torch.from_numpy(x_val).type(torch.Tensor).to(model.device)

    y_train_pred = model(x_train).detach().to(torch.device('cpu'))
    y_train_pred = scaler.inverse_transform(y_train_pred.numpy())
    y_train = scaler.inverse_transform(y_train)
    # plt.plot(range(y_train.shape[0]), y_train[:, 0])
    # plt.plot(range(y_train_pred.shape[0]), y_train_pred[:, 0])
    # plt.show()

    y_test_pred = model(x_test).detach().to(torch.device('cpu'))
    y_test_pred = scaler.inverse_transform(y_test_pred.numpy())
    y_test = scaler.inverse_transform(y_test)
    print(y_test_pred)
    print(y_test)

    y_val_pred = model(x_val).detach().to(torch.device('cpu'))
    y_val_pred = scaler.inverse_transform(y_val_pred.numpy())
    y_val = scaler.inverse_transform(y_val)

    trainMSE = mean_squared_error(y_train, y_train_pred)
    trainR2 = r2_score(y_train, y_train_pred)
    print('Train MSE:', trainMSE)
    print('Train R2:', trainR2)

    valMSE = mean_squared_error(y_val, y_val_pred)
    valR2 = r2_score(y_val, y_val_pred)
    print('Validation MSE:', valMSE)
    print('Validation R2:', valR2)

    testMSE = mean_squared_error(y_test, y_test_pred)
    testR2 = r2_score(y_test, y_test_pred)
    print('Test MSE:', testMSE)
    print('Test R2:', testR2)

    # data visualization
    fig, ax = plt.subplots(1,1)
    # fig = plt.Figure(figsize=(15, 6))
    # ax: plt.Axes = fig.subplots(1, 1)
    len_train, len_val, len_test = len(y_train), len(y_val), len(y_test)
    ax.plot(range(0, len_train), y_train[:, 0], 'b')
    ax.plot(range(0, len_train), y_train_pred[:, 0], 'darkorange')

    ax.plot(range(len_train, len_train + len_val), y_val[:, 0], 'b')
    ax.plot(range(len_train, len_train + len_val), y_val_pred[:, 0], 'darkorange')

    ax.plot(range(len_train + len_val, len_train + len_val + len_test), y_test[:, 0], 'b')
    ax.plot(range(len_train + len_val, len_train + len_val + len_test), y_test_pred[:, 0], 'darkorange')

    ax.axvline(x=len_train, ls='--', c='r')
    ax.axvline(x=len_train+len_val, ls='--', c='r')

    plt.show()


if __name__ == '__main__':
    MSE_lst = []
    drt = './noise/modelpara-time96'
    for filename in os.listdir(drt):
        if filename.endswith('.txt'):
            with open(f'{drt}/{filename}') as file:
                info = file.read()
                idx = info.find('MSE:')
                MSE_lst.append((filename[:-8], eval(info[idx + 5:])))
    MSE_lst.sort(key=lambda x: x[-1])
    print(MSE_lst)
    file = f'{drt}/{MSE_lst[0][0]}hyperpara.bin'
    # file = filedialog.askopenfilename()
    print('Check:', file)
    with open(file, 'rb') as f:
        para = pickle.load(f)
    test(para)
