# 交替训练     

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import ast

def create_sliding_window(data_x, data_y, window_size):
    X, y = [], []
    for i in range(len(data_x) - window_size-200):
        X.append(data_x[i:i + window_size])  # 获取窗口内的值
        y.append(data_y[i + window_size])     # 获取窗口外的目标值
    return torch.stack(X), torch.stack(y)

def train_alternating():
    num_epochs = 200
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()  # 切换到训练模式
        total_train_loss = 0

        # 获取两个迭代器
        iter1 = iter(train_loader1)
        iter2 = iter(train_loader2)

        # 交替使用两个数据加载器进行训练
        while True:
            try:
                # 从第一个时间序列数据加载器获取一批数据
                x_batch1, y_batch1 = next(iter1)
                # 从第二个时间序列数据加载器获取一批数据
                x_batch2, y_batch2 = next(iter2)

                # 计算第一个序列的预测和损失
                z1 = model(x_batch1)
                loss1 = loss_fn(z1, y_batch1)

                # 计算第二个序列的预测和损失
                z2 = model(x_batch2)
                loss2 = loss_fn(z2, y_batch2)

                # 计算总损失
                total_loss = loss1 + loss2
                total_train_loss += total_loss.item()

                # 反向传播
                 # 在这里进行反向传播

            except StopIteration:
                break  # 当其中一个加载器遍历完时，停止训练
        total_loss.backward()
        optimizer.step()  # 更新参数
        train_losses.append(total_train_loss)

         # 测试模型
        model.eval()  # 切换到评估模式
        total_test_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                z = model(x_batch)
                loss = loss_fn(z, y_batch)
                total_test_loss += loss.item()

        test_losses.append(total_test_loss * 5)

        # 打印每个 epoch 的损失
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {total_train_loss:.4f}, Test Loss: {total_test_loss:.4f}')

        optimizer.zero_grad()  # 在每个 epoch 开始时清除梯度

    torch.save(model.state_dict(), 'rnn_model.pth')
    
    plt.figure(figsize=(12, 6))
    # 绘制训练和测试损失曲线
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()
    plt.show()

    return


# 定义RNN的模型类
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 只取最后一个时间步的输出
        return out


# 实例化模型
input_size = 6
hidden_size = 30
output_size = 1
model = LSTMModel(input_size, hidden_size, output_size)

# 定义损失函数
loss_fn = nn.MSELoss()

# 定义优化器
weight_decay = 0
optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=weight_decay)

# 数据的读取
def str_to_list(s):
    return ast.literal_eval(s)

data = pd.read_csv('C:\\Users\\Lenovo\\OneDrive\\A_CodePython\\MachineLearingQC\\TimeSeriesPrediction\\NewDataM001.csv', converters={'矢量': str_to_list})

# 读取两个时间序列数据
x1 = torch.tensor(data['矢量'], dtype=torch.float32)  # 第一个序列的输入
y1 = torch.tensor(data['放电容量/Ah'], dtype=torch.float32)  # 第一个序列的目标

data = pd.read_csv('C:\\Users\\Lenovo\\OneDrive\\A_CodePython\\MachineLearingQC\\TimeSeriesPrediction\\NewDataM012.csv', converters={'矢量': str_to_list})

x2 = torch.tensor(data['矢量'], dtype=torch.float32)  # 第二个序列的输入
y2 = torch.tensor(data['放电容量/Ah'], dtype=torch.float32)  # 第二个序列的目标

# 创建滑动窗口的输入和目标
window_size = 5
x_windows1, y_windows1 = create_sliding_window(x1, y1, window_size)
x_windows2, y_windows2 = create_sliding_window(x2, y2, window_size)

# 划分训练集和测试集
x_train1, x_test1, y_train1, y_test1 = train_test_split(x_windows1, y_windows1, test_size=0.2, random_state=42)
x_train2, x_test2, y_train2, y_test2 = train_test_split(x_windows2, y_windows2, test_size=0.2, random_state=42)

# 创建数据集和数据加载器
train_dataset1 = TensorDataset(x_train1, y_train1)
test_dataset1 = TensorDataset(x_test1, y_test1)

train_dataset2 = TensorDataset(x_train2, y_train2)
test_dataset2 = TensorDataset(x_test2, y_test2)

train_loader1 = DataLoader(train_dataset1, batch_size=1, shuffle=True)
train_loader2 = DataLoader(train_dataset2, batch_size=1, shuffle=True)

test_loader = DataLoader(test_dataset1, batch_size=1, shuffle=False)  # 假设测试集相同

train_alternating() # 开始用总损失函数训练