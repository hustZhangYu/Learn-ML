# 这个代码用于数据的读取和测试
import re
from tokenize import group
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from torch import t
import numpy as np


def DataPlot():
    def DataPlotTest(data1,data2):
    # 画出曲线看看
        plt.plot(data1,data2)
    #  plt.show()

    # 读取数据

    z=['01','03','06','12','13','16','19']

    for z1 in z:
        data=pd.read_excel(fr'C:\Users\Lenovo\OneDrive\A_CodePython\MachineLearingQC\TimeSeriesPrediction\train\M0'+z1+'.xlsx')

    # 去掉第一行数据编号
        data=data.iloc[:,1:]
        DataPlotTest(data.iloc[:,0],data.iloc[:,1])
        print(data)

    plt.show()
    return 

def DealWithOutliers(c):
    data=pd.read_excel(fr'C:\Users\Lenovo\OneDrive\A_CodePython\MachineLearingQC\TimeSeriesPrediction\train\M0'+c+'.xlsx')
    data=data.iloc[:,1:]
    # print(data.describe())
    
    # data.boxplot(column='放电容量/Ah')    
    # plt.show()

    
    Q1=data['放电容量/Ah'].quantile(0.25)
    Q3=data['放电容量/Ah'].quantile(0.75)
    IQR=Q3-Q1
    lower_bound=Q1-1.5*IQR
    upper_bound=Q3+1.5*IQR

    ## 直接剔除异常值
    # plt.plot(data.iloc[:,0],data.iloc[:,1],'.',label='Original data')
    # data=data.loc[(data['放电容量/Ah']>=lower_bound) & (data['放电容量/Ah']<=upper_bound )]
    # plt.plot(data.iloc[:,0],data.iloc[:,1],label='New data')
    # plt.legend()
    # plt.show()

    # 将异常值使用合适的值替代
    # 直接剔除异常值可能不利于时间序列预测
    # 因此我们先将数据替代为NaN,然后通过插值来平滑过渡
    # 替换异常值为 NaN，再进行插值
    # plt.plot(data.iloc[:,0],data.iloc[:,1],'.',label='Original data')
    outliers = data[(data['放电容量/Ah'] <= lower_bound) | (data['放电容量/Ah'] >= upper_bound)]
    # 将异常值替换为 pd.NA
    data.loc[outliers.index, '放电容量/Ah'] = pd.NA
    data['放电容量/Ah'] = data['放电容量/Ah'].interpolate(method='linear', limit_direction='both')
    ## 关于画图
    # plt.plot(data.iloc[:,0],data.iloc[:,1],label='New data')
    # plt.legend()
    # plt.show()

    return data


def custom_minmax_scaler(data, new_min=0, new_max=1):
    data_min = np.min(data)
    data_max = np.max(data)
    
    scaled_data = new_min + (data - data_min) * (new_max - new_min) / (data_max - data_min)
    return scaled_data


def GeneratingInput1(window_size=5):
    # 处理数据和异常值
    data = DealWithOutliers('12')
    data1 = data['放电容量/Ah'].values.reshape(-1, 1)  # 将一维数据转成二维
        
    # 对整个 data 进行归一化
    new_min = 100
    new_max = 200
    data1_scaled= data1 .apply(lambda x: custom_minmax_scaler(x, new_min, new_max))
    # print(data1_scaled)
    
    
    # plt.plot(data1_scaled)
    # 使用滑动窗口生成 x 和 y
    x = []
    y = []
    for i in range(len(data1_scaled) - window_size):
        x.append(data1_scaled[i:i + window_size].tolist())  # 每个窗口的数据作为输入
        y.append(data1_scaled[i + window_size])  # 窗口结束后的数据作为输出
    
    # 创建包含 x 和 y 的 DataFrame
    df = pd.DataFrame({'x': x, 'y': y})
    
    # print(df)
    return df



def ReadInput(m):

    def time_to_seconds(t):
        return t.hour * 1 + t.minute / 60 + t.second/3600

    # def readfile(m):
    #     # we use this function to read 工步数据
    #     data=pd.read_excel(fr'C:\Users\Lenovo\OneDrive\A_CodePython\MachineLearingQC\TimeSeriesPrediction\train\M0'+m+'.xlsx',sheet_name='工步数据')

    #     # 对于工步数据表格中的“工步号”进行处理
    #     x=data['工步号']
    #     x1=(x-x.min())/(x.max()-x.min())
    #     data['工步号']=x1


    #     # 对 “工步状态”进行处理
    #     label_encoder = LabelEncoder()
    #     data['工步状态'] = label_encoder.fit_transform(data['工步状态'])
    #     # 归一化处理
    #     x=data['工步状态']
    #     x1=(x-x.min())/(x.max()-x.min())
    #     data['工步状态']=x1

    #     # 或许我们直接可以将工步号变成一个1*5的矢量，里面记录总的工作时间

    #     return x1
    
    def readdetails(m):
        # we use this function to read 详细数据

        def create_vector(df):
            # 创建一个1*5的矢量，初始值为0
            vector = [0] * 5
            for _, row in df.iterrows():
                # 工步号从1-5，填入到矢量的对应位置
                if 1 <= row['工步号'] <= 5:
                    vector[row['工步号'] - 1] = row['工步时间_h']
            return vector

        # 我们从详细数据中读取每一步的工作数据
        xls = pd.ExcelFile(fr'C:\Users\Lenovo\OneDrive\A_CodePython\MachineLearingQC\TimeSeriesPrediction\train\M0'+m+'.xlsx')
        sheet_names = xls.sheet_names

        # 筛选出以 "详细数据" 开头的 sheet 名称
        target_sheets = [sheet for sheet in sheet_names if sheet.startswith("详细数据")]

        # 初始化一个空的 DataFrame
        all_data = pd.DataFrame()

        # 循环读取每个符合条件的 sheet，并将数据整合
        for sheet in target_sheets:
            # 读取当前 sheet
            df = pd.read_excel(xls, sheet_name=sheet)
            
            # 将当前 sheet 的数据添加到 all_data 中
            all_data = pd.concat([all_data, df], ignore_index=True)

        # 读取流程时间
        changes = all_data['工步号'] != all_data['工步号'].shift(-1)

        selected_rows = all_data[changes]
        selected_rows['工步时间_h'] = selected_rows['工步时间'].apply(time_to_seconds)

        # 进一步处理
        result = pd.DataFrame(columns=['循环号', '矢量'])
        rows = []  # 用于暂存每个循环号的结果
        for cycle, group in selected_rows.groupby('循环号'):
            vector = create_vector(group)
            rows.append({'循环号': cycle, '矢量': vector})

        # 使用 pd.concat() 将所有结果整合到 result DataFrame 中
        result = pd.concat([result, pd.DataFrame(rows)], ignore_index=True)
        
        return result

    data=readdetails(m)

    return data       


def GeneratingInput(window_size=5):
    # 处理数据和异常值
    m='12'
    data = DealWithOutliers(m)
    data_column = data['放电容量/Ah'].values.reshape(-1, 1)  # 将其转为二维数组

    # 创建 MinMaxScaler 实例，指定归一化范围为 [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))

    # 对数据进行归一化
    data_normalized = scaler.fit_transform(data_column)

    # 将归一化后的数据再存回到 DataFrame 中
    data['放电容量/Ah'] = data_normalized

    dataDetails=ReadInput(m)

    data=pd.concat([data,dataDetails['矢量']],axis=1)

    print(data)
    data['矢量'] = data.apply(lambda row: row['矢量'] + [row['放电容量/Ah']], axis=1)

    filename='C:\\Users\\Lenovo\\OneDrive\\A_CodePython\\MachineLearingQC\\TimeSeriesPrediction\\NewDataM0'+m+'.csv'

 
    data.to_csv(filename)
    # print(df)
    return data

if __name__=='__main__':
    # DealWithOutliers()
    # ReadInput()
    import os
    print(os.getcwd())
    GeneratingInput()
    