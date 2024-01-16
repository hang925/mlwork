import os
import pickle
from datasource import DataSource
from model import LSTM
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from para import Para


def train(para):
    torch.manual_seed(para.seed)
    torch.cuda.manual_seed(para.seed)
    torch.cuda.manual_seed_all(para.seed)
    np.random.seed(para.seed)

    model = LSTM(input_dim=para.input_dim, hidden_dim=para.hidden_dim, output_dim=para.output_dim,
                 num_layers=para.num_layers, dropout=para.drop)

    device = model.device
    print(device)

    model = model.to(device)
    criterion = torch.nn.MSELoss().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=para.learning_rate)

    trainLoss = np.zeros(para.num_epochs)
    valLoss = np.zeros(para.num_epochs)
    testLoss = np.zeros(para.num_epochs)

    src = DataSource()
    src.load_scale_split(para.lookback, para.future)

    x_train, y_train, x_val, y_val, x_test, y_test, scaler = src.fetch(para.mode)

    x_train = torch.from_numpy(x_train).type(torch.Tensor).to(device)
    y_train = torch.from_numpy(y_train).type(torch.Tensor).to(device)

    x_val = torch.from_numpy(x_val).type(torch.Tensor).to(device)
    y_val = scaler.inverse_transform(y_val)

    x_test = torch.from_numpy(x_test).type(torch.Tensor).to(device)
    y_test = scaler.inverse_transform(y_test)

    bestEpoch, bestValMSE = 0, np.inf
    for t in range(para.num_epochs):
        y_train_pred = model(x_train)

        # y_val_pred = model(x_val).detach().to(torch.device('cpu'))
        # y_val_pred = scaler.inverse_transform(y_val_pred.numpy())
        # valMSE = mean_squared_error(y_val, y_val_pred)
        # if valMSE < bestValMSE:
        #     bestEpoch, bestValMSE = t, valMSE
        #     torch.save(obj=model.state_dict(), f=para.modelParaPath())
        #     with open(file=para.runLogPath(), mode='w') as f:
        #         f.write(f'bestEpoch: {bestEpoch}; best validation MSE: {bestValMSE}')
        #     with open(file=para.hyperParaPath(), mode='wb') as f:
        #         pickle.dump(para, f, pickle.HIGHEST_PROTOCOL)

        loss = criterion(y_train_pred, y_train)
        # print("Epoch:", t, "trainMSE:", loss.item(), "valMSE:", valMSE)
        y_train_pred1 = y_train_pred.detach().to(torch.device('cpu'))
        y_train_pred1 = scaler.inverse_transform(y_train_pred1.numpy())
        y_train1 = scaler.inverse_transform(y_train.to(torch.device('cpu')))
        trainMSE = mean_squared_error(y_train1, y_train_pred1)
        trainLoss[t] = trainMSE

        y_test_pred = model(x_test).detach().to(torch.device('cpu'))
        y_test_pred = scaler.inverse_transform(y_test_pred.numpy())
        testMSE = mean_squared_error(y_test, y_test_pred)
        testLoss[t] = testMSE

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    plt.plot(range(para.num_epochs), trainLoss)
    plt.plot(range(para.num_epochs), testLoss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train loss', 'test loss'])
    plt.show()


if __name__ == '__main__':
    para = Para(mode='std', hd=128, lr=0.005, drop=0.1, out=336, avg=False)
    train(para)
    # for md in ['std', 'noise']:
    #     for out in [96, 336]:
    #         for hd in [32, 64, 128, 256]:
    #             for lr in [0.001, 0.005, 0.01, 0.05, 0.1]:
    #                 for dr in [0.1, 0.5, 0.9]:
    #                     para = Para(mode=md, hd=hd, lr=lr, drop=dr, out=out, avg=False)
    #                     train(para)

    # # std 96
    # for seed in range(5):
    #     para = Para(hd=128, lr=0.01, drop=0.1, seed=seed, avg=True, out=96, mode='std')
    #     train(para)
    #
    # for seed in range(5):
    #     para = Para(hd=256, lr=0.005, drop=0.1, seed=seed, avg=True, out=96, mode='std')
    #     para.seed = seed
    #     train(para)
    #
    # for seed in range(5):
    #     para = Para(hd=64, lr=0.01, drop=0.1, seed=seed, avg=True, out=96, mode='std')
    #     para.seed = seed
    #     train(para)
    #
    # for seed in range(5):
    #     para = Para(hd=128, lr=0.005, drop=0.1, seed=seed, avg=True, out=96, mode='std')
    #     para.seed = seed
    #     train(para)
    #
    # for seed in range(5):
    #     para = Para(hd=32, lr=0.01, drop=0.1, seed=seed, avg=True, out=96, mode='std')
    #     train(para)
    #
    # # std 336
    # for seed in range(5):
    #     para = Para(hd=128, lr=0.01, drop=0.1, seed=seed, avg=True, out=336, mode='std')
    #     train(para)
    #
    # for seed in range(5):
    #     para = Para(hd=32, lr=0.05, drop=0.1, seed=seed, avg=True, out=336, mode='std')
    #     para.seed = seed
    #     train(para)
    #
    # for seed in range(5):
    #     para = Para(hd=128, lr=0.005, drop=0.1, seed=seed, avg=True, out=336, mode='std')
    #     para.seed = seed
    #     train(para)
    #
    # for seed in range(5):
    #     para = Para(hd=64, lr=0.005, drop=0.1, seed=seed, avg=True, out=336, mode='std')
    #     para.seed = seed
    #     train(para)
    #
    # for seed in range(5):
    #     para = Para(hd=64, lr=0.01, drop=0.1, seed=seed, avg=True, out=336, mode='std')
    #     train(para)
    #
    # # noise 96
    # for seed in range(5):
    #     para = Para(hd=64, lr=0.01, drop=0.5, seed=seed, avg=True, out=96, mode='noise')
    #     train(para)
    #
    # for seed in range(5):
    #     para = Para(hd=64, lr=0.01, drop=0.1, seed=seed, avg=True, out=96, mode='noise')
    #     para.seed = seed
    #     train(para)
    #
    # for seed in range(5):
    #     para = Para(hd=128, lr=0.01, drop=0.1, seed=seed, avg=True, out=96, mode='noise')
    #     para.seed = seed
    #     train(para)
    #
    # for seed in range(5):
    #     para = Para(hd=256, lr=0.01, drop=0.9, seed=seed, avg=True, out=96, mode='noise')
    #     para.seed = seed
    #     train(para)
    #
    # for seed in range(5):
    #     para = Para(hd=32, lr=0.05, drop=0.1, seed=seed, avg=True, out=96, mode='noise')
    #     train(para)
    #
    # # noise 336
    # for seed in range(5):
    #     para = Para(hd=128, lr=0.01, drop=0.9, seed=seed, avg=True, out=336, mode='noise')
    #     train(para)
    #
    # for seed in range(5):
    #     para = Para(hd=128, lr=0.005, drop=0.1, seed=seed, avg=True, out=336, mode='noise')
    #     para.seed = seed
    #     train(para)
    #
    # for seed in range(5):
    #     para = Para(hd=128, lr=0.005, drop=0.5, seed=seed, avg=True, out=336, mode='noise')
    #     para.seed = seed
    #     train(para)
    #
    # for seed in range(5):
    #     para = Para(hd=256, lr=0.005, drop=0.1, seed=seed, avg=True, out=336, mode='noise')
    #     para.seed = seed
    #     train(para)
    #
    # for seed in range(5):
    #     para = Para(hd=32, lr=0.01, drop=0.5, seed=seed, avg=True, out=336, mode='noise')
    #     train(para)
