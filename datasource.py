import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class DataSource:
    def __init__(self):
        self.scaler_x = MinMaxScaler(feature_range=(-1, 1))
        self.scaler_y = MinMaxScaler(feature_range=(-1, 1))

    def split(self, data, lookback=96, future=96):
        x_data = []
        y_data = []

        # 将data按lookback分组
        for index in range(len(data) - lookback - future):
            x_data.append(data[index: index + lookback])
            y_data.append(data[index + lookback: index + lookback + future, -1])
            # x_data.append(data_raw[index: index + lookback])
            # y_data.append(data_raw[index + 1: index + future + 1, -1])

        return np.array(x_data), np.array(y_data)

    def load_scale_split(self, lookback=96, future=96):
        raw_train = pd.read_csv('./data/train_set.csv').sort_values('date').loc[:, 'HUFL':]
        raw_val = pd.read_csv('./data/validation_set.csv').sort_values('date').loc[:, 'HUFL':]
        raw_test = pd.read_csv('./data/test_set.csv').sort_values('date').loc[:, 'HUFL':]

        self.scaler_x.fit(raw_train)
        self.scaler_y.fit(raw_train.loc[:, 'OT'].values.reshape(-1, 1))

        train_scaled = self.scaler_x.transform(raw_train)
        val_scaled = self.scaler_x.transform(raw_val)
        test_scaled = self.scaler_x.transform(raw_test)

        self.x_train, self.y_train = self.split(train_scaled, lookback, future)
        self.x_val, self.y_val = self.split(val_scaled, lookback, future)
        self.x_test, self.y_test = self.split(test_scaled, lookback, future)

        # for item in [self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test]:
        #     print(item.shape)


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
