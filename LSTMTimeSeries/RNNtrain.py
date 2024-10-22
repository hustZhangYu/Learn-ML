# 我们采用RNN模型来进行训练
# 滑动窗口办法

# 引入需要的库
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from DataSet import GeneratingInput, GeneratingInput1

# 定义多层感知机的模型类
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), hidden_size)  # 初始化隐藏状态
        out, _ = self.rnn(x, h_0)  # RNN 前向传播
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出，并通过全连接层得到预测
        return out

# 实例化模型，创建一个模型，给定输入层大小，隐藏层大小和输出层大小
input_size = 5
hidden_size = 30
output_size = 1


model = RNNModel(input_size, hidden_size, output_size)

# 定义损失函数
loss_fn = nn.MSELoss()
# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 引入数据
data = GeneratingInput()
# 第一列作为输入，第二列作为目标
x = torch.tensor(data.iloc[:, 0].tolist(), dtype=torch.float32)  # 第一列
y = torch.tensor(data.iloc[:, 1].values, dtype=torch.float32)  # 第二列

# 设置滑动窗口参数
window_size = 5  # 窗口大小
stride = 1        # 步幅

# 创建滑动窗口样本
x_windows = []
y_windows = []
for i in range(0, len(x) - window_size + 1, stride):
    x_windows.append(x[i:i + window_size])  # 提取窗口内的输入数据
    y_windows.append(y[i + window_size - 1])  # 目标是窗口结束时的值

# 转换为 Tensor
x_windows = torch.stack(x_windows)  # 形状为 (num_samples, window_size)
y_windows = torch.tensor(y_windows, dtype=torch.float32)  # 形状为 (num_samples,)

# 创建数据集和数据加载器
dataset = TensorDataset(x_windows, y_windows)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

print(data_loader)

# 训练模型
num_epochs = 200 # 迭代次数

for epoch in range(num_epochs):
    total_loss = 0
    for x_batch, y_batch in data_loader:
        # print(x_batch,y_batch)
        optimizer.zero_grad()
        z = model(x_batch)
        loss = loss_fn(z, y_batch)
        total_loss += loss.abs()  # 累加损失

    total_loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}')


# 切换模型为评估模式
model.eval()




# 引入新的数据模型看看效果
# 引入数据
data = GeneratingInput1()
# 第一列作为输入，第二列作为目标
x = torch.tensor(data.iloc[:, 0].tolist(), dtype=torch.float32)  # 第一列
y = torch.tensor(data.iloc[:, 1].values, dtype=torch.float32)  # 第二列

# 设置滑动窗口参数
window_size = 5  # 窗口大小
stride = 1        # 步幅

# 创建滑动窗口样本
x_windows = []
y_windows = []
for i in range(0, len(x) - window_size + 1, stride):
    x_windows.append(x[i:i + window_size])  # 提取窗口内的输入数据
    y_windows.append(y[i + window_size - 1])  # 目标是窗口结束时的值

# 转换为 Tensor
x_windows = torch.stack(x_windows)  # 形状为 (num_samples, window_size)
y_windows = torch.tensor(y_windows, dtype=torch.float32)  # 形状为 (num_samples,)

# 创建数据集和数据加载器
dataset = TensorDataset(x_windows, y_windows)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)



# 使用没有梯度计算的环境来进行预测
with torch.no_grad():
    # 准备测试数据（这里假设你想要用完整的原始数据进行预测）
    test_x = x_windows  # 这里可以换成你的测试集

    # 模型预测
    predicted = model(test_x)  # 模型预测
    predicted = predicted.squeeze().numpy()  # 将预测结果转换为 numpy 格式
    true_values = y_windows.numpy()  # 真实值转换为 numpy 格式

    # 绘制真实值与预测值对比
    # plt.figure()
    plt.plot(true_values, label='True Values', color='blue')
    plt.plot(predicted, label='Predicted Values', color='red')
    plt.title('True vs Predicted Values')
    plt.xlabel('Samples')
    plt.ylabel('Value')
    plt.legend()
    # plt.show()

    # 引入新的数据模型看看效果
# 引入数据
data = GeneratingInput()
# 第一列作为输入，第二列作为目标
x = torch.tensor(data.iloc[:, 0].tolist(), dtype=torch.float32)  # 第一列
y = torch.tensor(data.iloc[:, 1].values, dtype=torch.float32)  # 第二列

# 设置滑动窗口参数
window_size = 5  # 窗口大小
stride = 1        # 步幅

# 创建滑动窗口样本
x_windows = []
y_windows = []
for i in range(0, len(x) - window_size + 1, stride):
    x_windows.append(x[i:i + window_size])  # 提取窗口内的输入数据
    y_windows.append(y[i + window_size - 1])  # 目标是窗口结束时的值

# 转换为 Tensor
x_windows = torch.stack(x_windows)  # 形状为 (num_samples, window_size)
y_windows = torch.tensor(y_windows, dtype=torch.float32)  # 形状为 (num_samples,)

# 创建数据集和数据加载器
dataset = TensorDataset(x_windows, y_windows)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)



# 使用没有梯度计算的环境来进行预测
with torch.no_grad():
    # 准备测试数据（这里假设你想要用完整的原始数据进行预测）
    test_x = x_windows  # 这里可以换成你的测试集

    # 模型预测
    predicted = model(test_x)  # 模型预测
    predicted = predicted.squeeze().numpy()  # 将预测结果转换为 numpy 格式
    true_values = y_windows.numpy()  # 真实值转换为 numpy 格式

    # 绘制真实值与预测值对比
    #plt.figure()
    plt.plot(true_values, label='True Values', color='blue')
    plt.plot(predicted, label='Predicted Values', color='red')
    plt.title('True vs Predicted Values')
    plt.xlabel('Samples')
    plt.ylabel('Value')
    plt.legend()
    plt.show()