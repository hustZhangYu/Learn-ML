# 这个代码用于数据的读取和测试
import re
from tokenize import group
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from torch import t
import numpy as np

def rescale():
    # 对所有的数据进行收集并做归一化后，返回到各个文件当中
    # 针对每一个文件，读取相应的量
    # 合并之后，去掉异常点
    # 进行归一化
    c_values = ['01', '12', '13']  # 这里替换为实际编号
    file_data = {}
    param_values = []

    # 构建路径并读取文件
    for c in c_values:
        # 根据 c 构建文件路径
        filepath = fr'C:\Users\Lenovo\OneDrive\A_CodePython\MachineLearingQC\TimeSeriesPrediction\train\M0{c}.xlsx'
        
        # 读取 Excel 文件
        df = pd.read_excel(filepath)
        
        # 提取目标列 'param'，收集起来用于归一化
        param_values.extend(df['放电容量/Ah'].values)  # 将所有文件中的 param 数据收集起来
        file_data[filepath] = df  # 记录下每个文件的数据

    # 将所有的 param 数据合并到一起，并进行归一化
    scaler = MinMaxScaler()
    param_values = pd.DataFrame(param_values, columns=['放电容量/Ah'])  # 转为 DataFrame 格式
    scaler.fit(param_values)  # 基于所有文件的 param 进行归一化

    # 逐个文件应用归一化后的参数值，并保存到新列中
    for filepath, df in file_data.items():
        df['放电容量/Ah'] = scaler.transform(df[['放电容量/Ah']]) 
        # 将带有新列的 DataFrame 保存到 Excel 文件中，避免覆盖原始列
        new_filepath = filepath.replace('.xlsx', '_normalized.xlsx')  # 保存为新的文件
        df.to_excel(new_filepath, index=False)

    return


def DealWithOutliers(c):
    data=pd.read_excel(fr'C:\Users\Lenovo\OneDrive\A_CodePython\MachineLearingQC\TimeSeriesPrediction\train\M0'+c+'_normalized.xlsx')
    data=data.iloc[:,1:]
  
    Q1=data['放电容量/Ah'].quantile(0.25)
    Q3=data['放电容量/Ah'].quantile(0.75)
    IQR=Q3-Q1
    lower_bound=Q1-1.5*IQR
    upper_bound=Q3+1.5*IQR

    outliers = data[(data['放电容量/Ah'] <= lower_bound) | (data['放电容量/Ah'] >= upper_bound)]
    # 将异常值替换为 pd.NA
    data.loc[outliers.index, '放电容量/Ah'] = pd.NA
    data['放电容量/Ah'] = data['放电容量/Ah'].interpolate(method='linear', limit_direction='both')

    return data




def ReadInput(m):

    def time_to_seconds(t):
        return t.hour * 1 + t.minute / 60 + t.second/3600

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




def GeneratingInput(m):
    # 处理数据和异常值
    
    data = DealWithOutliers(m)
 
    # 创建 MinMaxScaler 实例，指定归一化范围为 [0, 1]
    data_column = data['放电容量/Ah'].values.reshape(-1, 1)
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

    data = data.iloc[:-3]
    data.to_csv(filename)
    # print(df)
    return data

if __name__=='__main__':
    # DealWithOutliers()
    # ReadInput()
    import os
    print(os.getcwd())
    rescale()
    GeneratingInput('13')
    