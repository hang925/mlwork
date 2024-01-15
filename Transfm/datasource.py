import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch


class DataSource:
    def __init__(self):
        self.scaler_x = MinMaxScaler(feature_range=(-1, 1))
        self.scaler_y = MinMaxScaler(feature_range=(-1, 1))
        self.data_raw = pd.DataFrame()
        self.data_scaled = pd.DataFrame()

    def load(self, filepath='../data/ETTh1.csv'):
        self.data_raw = pd.read_csv(filepath).sort_values('date')

    def scaling(self):
        temp = self.data_raw.loc[:, 'HUFL':]
        data_scaled_x = self.scaler_x.fit_transform(temp.loc[:, 'HUFL':'LULL'])
        data_scaled_y = self.scaler_y.fit_transform(temp.loc[:, 'OT'].values.reshape(-1, 1))
        self.data_scaled = np.concatenate([data_scaled_x, data_scaled_y], axis=1)

    def split(self, lookback=96, future=96):
        data_raw = self.data_scaled
        x_data = []
        y_data = []

        # 将data按lookback分组
        for index in range(len(data_raw) - lookback - future):
            x_data.append(data_raw[index: index + lookback])
            y_data.append(data_raw[index+lookback: index+lookback+future, -1])

        x_data = np.array(x_data)
        y_data = np.array(y_data)

        x_temp, x_test, y_temp, y_test = train_test_split(x_data, y_data, test_size=0.2, shuffle=False)
        x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.25, shuffle=False)

        self.x_train, self.y_train = x_train, y_train
        self.x_val , self.y_val = x_val, y_val
        self.x_test, self.y_test = x_test, y_test


    def load_scale_split(self, lookback=96, future=96):
        self.load()
        self.scaling()
        self.split(lookback, future)

if __name__ == '__main__':
    source = DataSource()
    source.load()
    raw_data = source.data_raw
    source.scaling()

    source.split()

    print(source.x_train)

    # plt.figure(figsize=(15, 9))
    # plt.plot(raw_data['OT'])
    # plt.xticks(range(0, raw_data.shape[0], 1000), raw_data['date'].loc[::1000], rotation=30)
    # plt.title("Oil Temperature", fontsize=18, fontweight='bold')
    # plt.xlabel('date', fontsize=18)
    # plt.ylabel('Temperature', fontsize=18)
    # plt.show()

