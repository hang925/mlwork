from typing import List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import ndarray, dtype
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class DataSource:
    def __init__(self):
        self.scaler_x = MinMaxScaler(feature_range=(-1, 1))
        self.scaler_ystd = MinMaxScaler(feature_range=(-1, 1))
        self.scaler_ynoise = MinMaxScaler(feature_range=(-1, 1))
        self.data_raw = pd.DataFrame()
        self.data_scaled = pd.DataFrame()

    def load(self, filepath='./data/ETTh1.csv'):
        self.data_raw = pd.read_csv(filepath).sort_values('date')

    def scaling(self):
        data_scaled_x = self.scaler_x.fit_transform(self.data_raw.loc[:, 'HUFL':'LULL'])
        x_std = np.arange(len(data_scaled_x)).reshape(-1, 1)
        data_scaled_ystd = self.scaler_ystd.fit_transform(self.data_std.reshape(-1, 1))
        data_scaled_ynoise = self.scaler_ynoise.fit_transform(self.data_noise.reshape(-1, 1))

        self.data_scaled_std = np.concatenate([x_std, data_scaled_ystd], axis=1)
        self.data_scaled_noise = np.concatenate([data_scaled_x, data_scaled_ynoise], axis=1)

    def split(self, x_data, y_std, y_noise, lookback=96, future=96):
        x_std = np.arange(len(x_data)).reshape(-1, 1)
        data_std = np.concatenate([x_std, y_std], axis=1)
        data_noise = np.concatenate([x_data, y_noise], axis=1)

        x_std_data = []
        y_std_data = []
        x_noise_data = []
        y_noise_data = []

        # 将data按lookback分组
        for index in range(len(data_std) - lookback - future):
            x_std_data.append(data_std[index: index + lookback, -1].reshape(-1, 1))
            y_std_data.append(data_std[index+lookback: index+lookback+future, -1])
            x_noise_data.append(data_noise[index: index + lookback])
            y_noise_data.append(data_noise[index + lookback: index + lookback + future, -1])

        x_std_data = np.array(x_std_data)
        y_std_data = np.array(y_std_data)
        x_noise_data = np.array(x_noise_data)
        y_noise_data = np.array(y_noise_data)

        return x_std_data, y_std_data, x_noise_data, y_noise_data

    def denoise(self, data):
        # data = np.array(self.data_raw.loc[:, 'OT'])
        # print(data)
        top, bot = data.copy(), data.copy()
        last_top, last_bot = 0, 0
        window = len(data) // 100
        for i in range(window, len(data) - window):
            if np.all(data[i - window: i + window + 1] - data[i] <= 0):
                top[last_top + 1: i] = np.linspace(top[last_top], top[i], num=i - last_top + 1)[1: -1]
                last_top = i
            elif np.all(data[i - window: i + window + 1] - data[i] >= 0):
                bot[last_bot + 1: i] = np.linspace(bot[last_bot], bot[i], num=i - last_bot + 1)[1: -1]
                last_bot = i
            else:
                pass
        top[last_top + 1: -1] = np.linspace(top[last_top], top[-1], num=len(data) - last_top)[1: -1]
        bot[last_bot + 1: -1] = np.linspace(bot[last_bot], bot[-1], num=len(data) - last_bot)[1: -1]
        mid = (top + bot) / 2
        mid[: window] = np.linspace(np.mean(data[: window]), mid[window], num=window + 2)[1: -1]
        mid[: -window - 1: -1] = np.linspace(mid[-window], np.mean(data[: -window: -1]), num=window + 2)[1: -1]
        noise = data - mid

        fig, ax = plt.subplots(1, 1)
        ax.plot(range(len(data)), data, 'b')
        ax.plot(range(len(data)), mid, 'r')
        ax.plot(range(len(data)), top, '--', c='darkorange')
        ax.plot(range(len(data)), bot, '--', c='darkorange')
        ax.set_xlabel('t')
        ax.set_ylabel('Oil Temperature')
        ax.legend(['raw data', 'medium'])
        plt.show()
        plt.plot(range(len(data)), noise, 'b')
        plt.show()

        # self.data_std, self.data_noise = mid, noise
        return mid, noise

    def load_scale_split(self, lookback=96, future=96):
        # self.load()
        # self.denoise()
        # self.scaling()
        # self.split(lookback, future)

        raw_train = pd.read_csv('./data/train_set.csv').sort_values('date').loc[:, 'HUFL':]
        raw_val = pd.read_csv('./data/validation_set.csv').sort_values('date').loc[:, 'HUFL':]
        raw_test = pd.read_csv('./data/test_set.csv').sort_values('date').loc[:, 'HUFL':]

        train_std, train_noise = self.denoise(np.array(raw_train.loc[:, 'OT']))
        val_std, val_noise = self.denoise(np.array(raw_val.loc[:, 'OT']))
        test_std, test_noise = self.denoise(np.array(raw_test.loc[:, 'OT']))

        # fig, ax = plt.subplots(1, 1)
        # l1, l2, l3 = 8640, 2976, 2976
        # ax.plot(range(l1), raw_train.loc[:, 'OT'], 'b')
        # ax.plot(range(l1), train_std, 'r')
        # plt.plot(range(l1), train_noise, '--k')
        # # ax.plot(range(l2), raw_test.loc[:, 'OT'], 'b')
        # # ax.plot(range(l2), test_std, 'r')
        # # plt.plot(range(l2), test_noise, '--k')
        # # ax.plot(range(l2), raw_val.loc[:, 'OT'], 'b')
        # # ax.plot(range(l2), val_std, 'r')
        # # plt.plot(range(l2), val_noise, '--k')
        # # ax.plot(range(len(data)), bot, 'r')
        # # ax.plot(range(len(data)), mid, 'darkorange')
        # # plt.show()
        # plt.show()

        self.scaler_x.fit(raw_train.loc[:, 'HUFL':'LULL'])
        self.scaler_ystd.fit(train_std.reshape(-1, 1))
        self.scaler_ynoise.fit(train_noise.reshape(-1, 1))

        train_x_scaled = self.scaler_x.transform(raw_train.loc[:, 'HUFL':'LULL'])
        train_std_scaled = self.scaler_ystd.transform(train_std.reshape(-1, 1))
        train_noise_scaled = self.scaler_ynoise.transform(train_noise.reshape(-1, 1))

        val_x_scaled = self.scaler_x.transform(raw_val.loc[:, 'HUFL':'LULL'])
        val_std_scaled = self.scaler_ystd.transform(val_std.reshape(-1, 1))
        val_noise_scaled = self.scaler_ynoise.transform(val_noise.reshape(-1, 1))

        test_x_scaled = self.scaler_x.transform(raw_test.loc[:, 'HUFL':'LULL'])
        test_std_scaled = self.scaler_ystd.transform(test_std.reshape(-1, 1))
        test_noise_scaled = self.scaler_ynoise.transform(test_noise.reshape(-1, 1))

        self.x_std_train, self.y_std_train, self.x_noise_train, self.y_noise_train \
            = self.split(train_x_scaled, train_std_scaled, train_noise_scaled, lookback, future)
        self.x_std_val, self.y_std_val, self.x_noise_val, self.y_noise_val \
            = self.split(val_x_scaled, val_std_scaled, val_noise_scaled, lookback, future)
        self.x_std_test, self.y_std_test, self.x_noise_test, self.y_noise_test \
            = self.split(test_x_scaled, test_std_scaled, test_noise_scaled, lookback, future)


    def fetch(self, mode):
        if mode == 'std':
            return (self.x_std_train, self.y_std_train,
                    self.x_std_val, self.y_std_val,
                    self.x_std_test, self.y_std_test,
                    self.scaler_ystd)
        else:
            return (self.x_noise_train, self.y_noise_train,
                    self.x_noise_val, self.y_noise_val,
                    self.x_noise_test, self.y_noise_test,
                    self.scaler_ynoise)


if __name__ == '__main__':
    source = DataSource()
    source.load_scale_split()

    # plt.figure(figsize=(15, 9))
    # plt.plot(raw_data['OT'])
    # plt.xticks(range(0, raw_data.shape[0], 1000), raw_data['date'].loc[::1000], rotation=30)
    # plt.title("Oil Temperature", fontsize=18, fontweight='bold')
    # plt.xlabel('date', fontsize=18)
    # plt.ylabel('Temperature', fontsize=18)
    # plt.show()