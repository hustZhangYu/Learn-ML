from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from DataSet import GeneratingInput, GeneratingInput1
import ast 
import pandas as pd 
from sklearn.metrics import mean_squared_error

# 定义LSTM的模型类
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

model = LSTMModel(input_size, hidden_size, output_size)  # 确保使用与保存时相同的参数
# 加载模型的状态字典
model.load_state_dict(torch.load('rnn_model.pth'))
model.eval()

# 数据的读取
def str_to_list(s):
    return ast.literal_eval(s)

data = pd.read_csv('C:\\Users\\Lenovo\\OneDrive\\A_CodePython\\MachineLearingQC\\TimeSeriesPrediction\\NewDataM013.csv', converters={'矢量': str_to_list})

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
data_loader = DataLoader(dataset, batch_size=5, shuffle=True)


# Use the last sliding window as input for prediction
test_size = 2180  # You can adjust this value based on your needs
x_test = x_windows[-test_size:]  # Get the last test_size windows
y_test = y_windows[-test_size:]  # Get the corresponding true target values

# Convert data to a shape suitable for model input
with torch.no_grad():
    predictions = model(x_test).squeeze()  # Make predictions and remove extra dimensions

# Convert predictions and true values to numpy arrays for plotting
predictions = predictions.numpy()
y_test = y_test.numpy()

# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error (MSE): {mse}")


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


# Add title and labels
plt.title('Comparison of Original Data, True Values, and Predicted Values')
plt.xlabel('Sample Index')
plt.ylabel('Discharge Capacity / Ah')
plt.legend()
plt.grid(True)
plt.show()