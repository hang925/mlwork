import os
import sys


class Para:
    def __init__(self, hd=0, lr=0, drop=0, out=96, seed=1, avg=False, mode='std'):
        self.seed = seed
        # 输入的维度
        self.input_dim = 1 if mode == 'std' else 7
        # 隐藏层特征的维度
        self.hidden_dim = hd
        # 循环的layers
        self.num_layers = 2
        self.lookback = 96
        self.output_dim = out
        self.future = out
        self.num_epochs = 100
        self.learning_rate = lr
        self.drop = drop
        self.avg = avg
        self.mode = mode

        # self.directory = f'./modelpara-drop'
        if avg:
            self.directory = f'./{mode}/average-time{out}/hd{self.hidden_dim}lr{self.learning_rate}drop{self.drop}'
        else:
            self.directory = f'./{mode}/modelpara-time{out}'
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def modelParaPath(self):
        if not self.avg:
            return f'{self.directory}/hd{self.hidden_dim}lr{self.learning_rate}drop{self.drop}.pth'
        else:
            return f'{self.directory}/seed{self.seed}.pth'

    def runLogPath(self):
        if not self.avg:
            return f'{self.directory}/hd{self.hidden_dim}lr{self.learning_rate}drop{self.drop}info.txt'
        else:
            return f'{self.directory}/seed{self.seed}info.txt'

    def hyperParaPath(self):
        if not self.avg:
            return f'{self.directory}/hd{self.hidden_dim}lr{self.learning_rate}drop{self.drop}hyperpara.bin'
        else:
            return f'{self.directory}/seed{self.seed}hyperpara.bin'


if __name__ == '__main__':
    MSE_lst = []
    for filename in os.listdir('./modelpara-1'):
        if filename.endswith('.txt'):
            with open(f'./modelpara-1/{filename}') as file:
                info = file.read()
                idx = info.find('MSE:')
                MSE_lst.append((filename[:-8], eval(info[idx + 5:])))
    MSE_lst.sort(key=lambda x: x[-1])
    print(MSE_lst)
