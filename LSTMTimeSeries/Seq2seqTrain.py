# 我们采用LSTM方法来进行拟合

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split  # 用于划分数据集
from DataSet import GeneratingInput
import pandas as pd
import ast 


def train1():

# 训练和测试模型，并记录损失
    num_epochs = 100
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()  # 切换到训练模式
        total_train_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            z = model(x_batch)
            loss = loss_fn(z, y_batch)
            total_train_loss += loss
        total_train_loss.backward()
        optimizer.step()
        train_losses.append(total_train_loss.item())

        # 测试模型
        model.eval()  # 切换到评估模式
        total_test_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                z = model(x_batch)
                loss = loss_fn(z, y_batch)
                total_test_loss += loss.item()
        
        
        test_losses.append(total_test_loss*5)

        # 打印每个 epoch 的损失
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {total_train_loss:.4f}, Test Loss: {total_test_loss:.4f}')

    torch.save(model.state_dict(), 'rnn_model.pth') 

    # 绘制训练和测试损失曲线
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()
    plt.show()

    return 

# 第二种训练模式


def train2():

# 训练和测试模型，并记录损失
    num_epochs = 100
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()  # 切换到训练模式
        total_train_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            z = model(x_batch)
            loss = loss_fn(z, y_batch)
            loss.backward()
            total_train_loss=total_train_loss+loss.item()
            optimizer.step()
        train_losses.append(total_train_loss)

        # 测试模型
        model.eval()  # 切换到评估模式
        total_test_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                z = model(x_batch)
                loss = loss_fn(z, y_batch)
                total_test_loss += loss.item()
        
        
        test_losses.append(total_test_loss*5)

        # 打印每个 epoch 的损失
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {total_train_loss:.4f}, Test Loss: {total_test_loss:.4f}')

    torch.save(model.state_dict(), 'rnn_model.pth') 

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

# 实例化模型，创建一个模型，给定输入层大小，隐藏层大小和输出层大小
input_size = 6
hidden_size = 30
output_size = 1


model = LSTMModel(input_size, hidden_size, output_size)

# 定义损失函数

loss_fn = nn.SmoothL1Loss()
# loss_fn = nn.MSELoss()    
# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.1)
#  动态调整学习率
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)


# 数据的读取
def str_to_list(s):
    return ast.literal_eval(s)
data = pd.read_csv('NewdataM001.csv', converters={'矢量': str_to_list})


# 第一列作为输入，第二列作为目标
x = torch.tensor(data['矢量'], dtype=torch.float32)  # 第一列
y = torch.tensor(data['放电容量/Ah'], dtype=torch.float32)  # 第二列



# 创建滑动窗口的输入和目标
def create_sliding_window(data_x, data_y, window_size):
    X, y = [], []
    for i in range(len(data_x) - window_size-200):
        X.append(data_x[i:i + window_size])  # 获取窗口内的值
        y.append(data_y[i + window_size])     # 获取窗口外的目标值
    return torch.stack(X), torch.stack(y)

# 设置窗口大小
window_size = 5
x_windows, y_windows = create_sliding_window(x, y, window_size)

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x_windows, y_windows, test_size=0.2, random_state=42)

# 创建数据集和数据加载器
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


train2()


