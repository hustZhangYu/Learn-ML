# 此处，我们将进一步优化RNN模型。具体的可能修改将包括以下方面：
# 1. 损失函数的定义
# 2. 正则化的引入
# 3. 更多参数的引入

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from DataSet import GeneratingInput

# 定义RNN的模型类
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size  # 保存隐藏层大小

    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), self.hidden_size)  # 初始化隐藏状态
        out, _ = self.rnn(x, h_0)  # RNN 前向传播
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出，并通过全连接层得到预测
        return out

# 实例化模型，创建一个模型，给定输入层大小，隐藏层大小和输出层大小
input_size = 6
hidden_size = 30
output_size = 1


model = RNNModel(input_size, hidden_size, output_size)

# 定义损失函数
loss_fn = nn.MSELoss()    
# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.01)
#  动态调整学习率
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

# 引入数据
data = GeneratingInput()
# 第一列作为输入，第二列作为目标
x = torch.tensor(data['矢量'], dtype=torch.float32)  # 第一列
y = torch.tensor(data['放电容量/Ah'], dtype=torch.float32)  # 第二列



# 创建滑动窗口的输入和目标
def create_sliding_window(data_x, data_y, window_size):
    X, y = [], []
    for i in range(len(data_x) - window_size):
        X.append(data_x[i:i + window_size])  # 获取窗口内的值
        y.append(data_y[i + window_size])     # 获取窗口外的目标值
    return torch.stack(X), torch.stack(y)

# 设置窗口大小
window_size = 5
x_windows, y_windows = create_sliding_window(x, y, window_size)



# 创建数据集和数据加载器
dataset = TensorDataset(x_windows, y_windows)  # 添加特征维度
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)


# 训练模型
num_epochs = 2000 # 迭代次数

for epoch in range(num_epochs):
    total_loss=0
    for x_batch, y_batch in data_loader:
        optimizer.zero_grad()
        z = model(x_batch)
        loss = loss_fn(z, y_batch)
        total_loss +=loss
    
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
# Validate the model
# Set the model to evaluation mode
model.eval()

# Use the last sliding window as input for prediction
test_size = 800  # You can adjust this value based on your needs
x_test = x_windows[-test_size:]  # Get the last test_size windows
y_test = y_windows[-test_size:]  # Get the corresponding true target values

# Convert data to a shape suitable for model input
with torch.no_grad():
    predictions = model(x_test).squeeze()  # Make predictions and remove extra dimensions

# Convert predictions and true values to numpy arrays for plotting
predictions = predictions.numpy()
y_test = y_test.numpy()

# Original data
original_y = y.numpy()

# Plot original data, true values, and predictions
plt.figure(figsize=(12, 6))

# Plot original data
plt.plot(original_y, label='Original Data', color='gray', alpha=0.5)

# Plot true values
plt.plot(range(len(original_y) - test_size, len(original_y)), y_test, label='True Values', marker='o', color='blue')

# Plot predicted values
plt.plot(range(len(original_y) - test_size, len(original_y)), predictions, label='Predicted Values', marker='x', color='red')

# Set y-axis limits
plt.ylim(0, original_y.max())  # y-axis limits from 0 to maximum value

# Add title and labels
plt.title('Comparison of Original Data, True Values, and Predicted Values')
plt.xlabel('Sample Index')
plt.ylabel('Discharge Capacity / Ah')
plt.legend()
plt.grid(True)
plt.show()