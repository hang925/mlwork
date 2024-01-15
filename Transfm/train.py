import torch
import torch.nn as nn
import numpy as np
import time
from matplotlib import pyplot as plt
from Transfm.get_data import *
from Transformer_model import TransAm
import math
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)
np.random.seed(0)

import warnings

batch_size = 96  # 批次大小
warnings.filterwarnings('ignore')

def train(train_data):
    model.train()
    total_loss = 0.
    start_time = time.time()

    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        data, targets = get_batch(train_data, i, batch_size)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()

        total_loss += loss.item()
        log_interval = int(len(train_data) /batch_size/ 5)
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data)//batch_size, scheduler.get_lr()[0],
                              elapsed * 1000 / log_interval,
                cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def plot_and_loss(eval_model, data_source, epoch):
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(data_source) - 1):
            data, target = get_batch(data_source, i, 1)
            output = eval_model(data)
            # print(output)
            total_loss += criterion(output, target).item()
            test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0)
            truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)

    # test_result = test_result.cpu().numpy() -> no need to detach stuff..
    # len(test_result)

    plt.plot(test_result, color="red")
    plt.plot(truth[:500], color="blue")
    plt.plot(test_result - truth, color="green")
    plt.grid(True, which='both')
    plt.axhline(y=0, color='k')
    plt.savefig('graph/transformer-epoch%d.png' % epoch)
    plt.close()

    return total_loss / i

# predict the next n steps based on the input data
def predict_future(eval_model, data_source, steps):
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    data, _ = get_batch(data_source, 0, 1)
    with torch.no_grad():
        for i in range(0, steps):
            output = eval_model(data[-96:])
            data = torch.cat((data, output[-1:]))

    data = data.cpu().view(-1)
    return data[-steps:]
    # # I used this plot to visualize if the model pics up any long therm struccture within the data.
    plt.plot(data,color="red")
    plt.plot(data[:input_window],color="blue")
    plt.grid(True, which='both')
    plt.axhline(y=0, color='k')
    plt.savefig('graph/transformer-future%d.png'%steps)
    plt.close()


#  测试部分
def evaluate(eval_model, data_source):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    eval_batch_size = 1000
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i,eval_batch_size)
            output = eval_model(data)
            total_loss += len(data[0])* criterion(output, targets).cpu().item()
    return total_loss / len(data_source)



model = TransAm().to(device)
criterion = nn.MSELoss()
lr = 0.005
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
# 初始化正无穷大
best_val_loss = float("inf")
# The number of epochs
epochs = 20
best_model = None

# 获取数据
# 获取数据
train_losses = []
test_losses = []
train_data, test_data, scaler = get_data('train_set.csv', 'test_set.csv', 'validation_set.csv')

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(train_data)
    train_loss = evaluate(model, train_data)
    train_losses.append(train_loss)
    # if(epoch % 10 is 0):
    #     val_loss = plot_and_loss(model, test_data, epoch)
    #     predict_future(model, test_data, 200)
    # else:
    #     val_loss = evaluate(model, test_data)
    test_loss = evaluate(model, test_data)
    test_losses.append(test_loss)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.5f} | test ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     test_loss, math.exp(test_loss)))
    print('-' * 89)

    # if val_loss < best_val_loss:
    #    best_val_loss = val_loss
    #    best_model = model
    scheduler.step()
# 保存模型
torch.save(model.state_dict(), "./save/pretrain.pt")

# 保存损失值
df_loss = pd.DataFrame()
df_loss["epoch"] = list(range(1, epochs + 1))
df_loss["train loss"] = train_losses
df_loss["test loss"] = test_losses
df_loss.to_excel("./save/flu_loss.xlsx", index=None)

# 预测
pred = predict_future(model, train_data, 162)
actual_predictions = np.array(pred)
unscaled = [int(i) for i in list(scaler.inverse_transform(actual_predictions.reshape(-1, 1)).reshape(-1))]
df_pred = pd.DataFrame()
df_pred["prediction"] = unscaled
df_pred.to_excel("../save/flu_pred.xlsx", index=None)


