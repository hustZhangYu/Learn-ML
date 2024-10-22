# 这个文件中，我们采用MPL，多层感知机来进行时间序列预测
# 我们首先采用单变量 MLP 模型

# 引入需要的库
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from DataSet import GeneratingInput


# 定义多层感知机的模型类  简单起见，我们只带搭建了一层
class MLP(nn.Module):
    def __init__(self, input_size,hidden_size,output_size):
        super(MLP,self).__init__()  # 调用父类的__init__()
        self.fc1=nn.Linear(input_size,hidden_size)
        self.relu=nn.ReLU()
        self.fc2=nn.Linear(hidden_size,hidden_size)
        self.relu1=nn.ReLU()
        self.fc3=nn.Linear(hidden_size,output_size)

    def forward(self,x):
        # forward 函数定义了数据在模型中的流动方式
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu1(x)
        x = self.fc3(x)
        return x

# 实例化模型，创建一个模型，给定输入层大小， 隐藏层大小和输出层大小
input_size=5
hidden_size=30
output_size=1

model=MLP(input_size,hidden_size,output_size)
   
# 定义损失函数
loss_fn=nn.MSELoss()
# 定义优化器
optimizer=optim.Adam(model.parameters(),lr=0.0001) 


#引入数据
data=GeneratingInput()
x = torch.tensor(data.iloc[:, 0].tolist(), dtype=torch.float32)  # 第一列作为输入
y = torch.tensor(data.iloc[:, 1].values, dtype=torch.float32)  # 第二列作为目标
dataset = TensorDataset(x, y)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 训练模型
num_epochs=2000 #迭代次数

for epoch in range(num_epochs):
    total_loss = 0
    for x_batch, y_batch in data_loader:
        optimizer.zero_grad()
        z=model(x_batch)
        loss=loss_fn(z,y_batch)
        total_loss += loss.abs()

    total_loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item():.4f}')
     

# 先拿原本的数据集进行测试
first_row = next(iter(data_loader))
x_first = first_row[0][0]  
y_first = first_row[1][0]
print(x_first)
x_first=torch.cat((x_first[1:5],y_first.unsqueeze(0)))
print(x_first)

predictions = []
num_predictions = 1000 
for _ in range(num_predictions):
     # 将输入数据传入模型进行预测
    with torch.no_grad():  # 不需要计算梯度
        prediction = model(x_first.unsqueeze(0))  # 添加 batch 维度    
        predictions.append(prediction)
    # 更新输入数据为预测结果（根据你的模型和需求）
    
    x_first=torch.cat((x_first[1:5],prediction.squeeze(0) )) # 将预测结果作为下一次的输入

# 将预测结果转换为 NumPy 数组或其他格式
predictions = torch.cat(predictions).numpy() 
plt.plot(predictions)
plt.show()