# 我们采用LSTM方法来进行拟合
# 这种目标函数，首先定义太复杂，其次泛化能力太弱

from matplotlib import pyplot as plt  
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from DataSet import GeneratingInput
import ast 
import pandas as pd 

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
# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.1)
#  动态调整学习率
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.01)

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

print(y_windows.size())

# 创建数据集和数据加载器
dataset = TensorDataset(x_windows, y_windows)  # 添加特征维度
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)


print(data_loader)

# 训练模型
num_epochs = 100 # 迭代次数

for epoch in range(num_epochs):
    total_loss=0

    x_test = x_windows[0].unsqueeze(0) 
    for step in range(len(y_windows)):
        optimizer.zero_grad()
        z = model(x_test)
        y_batch=y_windows[step].unsqueeze(0)
        loss = loss_fn(z, y_batch)
        total_loss +=loss

        # 获取已知的输入量 (x)，并将其转换为张量
        x1 = x[window_size + step].clone()  # 克隆当前已知量以防止影响原始数据
        x1[-1] = z.item()  # 替换最后一个特征为新的预测结果

        # 更新窗口：前移并替换最后一个元素
        # 克隆 x_test 以避免内存冲突
        updated_x_test = x_test.clone()  
        updated_x_test[:, :-1, :] = x_test[:, 1:, :]  # 前移窗口，去掉第一个时间步
        updated_x_test[:, -1, :] = x1.unsqueeze(0)  # 将新的已知输入放入窗口的最后一个位置
        # 更新 x_test 为新的张量
        x_test = updated_x_test
    
    total_loss.backward()
    optimizer.step()
    scheduler.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}')

# for epoch in range(num_epochs):
#     for x_batch, y_batch in data_loader:
#         optimizer.zero_grad()
#         z = model(x_batch)
#         loss = loss_fn(z, y_batch)
#         loss.backward()
#         optimizer.step()
#     scheduler.step()

#     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}')

torch.save(model.state_dict(), 'rnn_model.pth') 