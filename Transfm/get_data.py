import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

input_window = 96  # number of input steps
output_window = 1  # number of prediction steps, in this model its fixed to one
batch_size = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + output_window:i + tw + output_window]
        inout_seq.append((train_label, train_seq))
    return torch.FloatTensor(inout_seq)

def get_data(traindata, testdata, validationdata):
    train_data = pd.read_csv("../data/" + traindata)
    train_amplitude = train_data["HUFL"].values
    train_data_length = len(train_amplitude)
    print("data shape: ", train_amplitude.shape)
    print("data length: ", train_data_length)

    test_data = pd.read_csv("../data/" + testdata)
    test_amplitude = test_data["HUFL"].values
    test_data_length = len(test_amplitude)
    print("data shape: ", test_amplitude.shape)
    print("data length: ", test_data_length)

    validation_data = pd.read_csv("../data/" + validationdata)
    validation_amplitude = validation_data["HUFL"].values
    validation_data_length = len(validation_amplitude)
    print("data shape: ", validation_amplitude.shape)
    print("data length: ", validation_data_length)

    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_amplitude_scale = scaler.fit_transform(train_amplitude.reshape(-1, 1)).reshape(-1)
    test_amplitude_scale = scaler.fit_transform(test_amplitude.reshape(-1, 1)).reshape(-1)
    validation_amplitude_scale = scaler.fit_transform(validation_amplitude.reshape(-1, 1)).reshape(-1)

    train_data = train_amplitude_scale[:int(train_data_length)]
    test_data = test_amplitude_scale[:int(test_data_length)]
    val_data = validation_amplitude_scale[:int(validation_data_length)]

    print("train_data shape：")
    print(train_data.shape)
    print("test_data shape：")
    print(test_data.shape)
    print("val_data shape：")
    print(val_data.shape)

    # 获取seq2seq数据，如1-100天为src，2-101天为tgt
    train_sequence = create_inout_sequences(train_data, input_window)
    test_sequence = create_inout_sequences(test_data, input_window)
    print("train sequence shape：")
    print(train_sequence.shape)
    print("test sequence shape：")
    print(test_sequence.shape)

    return train_sequence.to(device), test_sequence.to(device), scaler


def get_batch(source, i, batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i + seq_len]
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window, 1))  # 1 is feature size
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 1))
    return input, target


if __name__ == '__main__':
    get_data('train_set.csv', 'test_set.csv', 'validation_set.csv')
