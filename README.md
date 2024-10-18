# Learn-ML
People learn Machine learning. 

## ML to perdict the time evolution
First, we use the time series to predict the time evolution of the system.
Why people always study the time evolution of the open systems rather than isolated system?

## ML in disorder system

## 时间序列预测

非常幸运能有一个机会来上手实操一个机器学习问题。
简单描述一下题目： 结合数据及系统运行数据中工步号、工步状态、运行时间等各变量与电池放电容量之间的动态变化，同时也可以使用温度、气压等外部数据，构建电池放电容量预测算法模型，预测一段时间内电池放电容量随循环号变化的趋势。

#### 1. 数据导入和相关性统计
我们首先录入label。 其次我们引入相应的因变量。但哪些因变量会影响系统的时间演化？我们首先进行相关性统计，考虑这些因变量是否真的与标签相关。
我们首先导入工步号，公布状态和运行时间。
#### 2. 数据预处理和归一化
这一部分，我们主要考虑如何对数据进行一个预先的处理，包括识别并且修改一些异常值，对数据进行归一化，方便之后的训练。
#### 3. 多层感知机（MLP）训练
我们尝试用最经典的多层感知机MLP来预测时间序列。然而实际情况表明，效果不太好。
#### 4. 迭代神经网络（RNN）训练
我们首先训练单参数输入的神经网络来对时间序列进行预测。
首先介绍RNN的基本原理：
#### 5. 更多参数的引入和模型的泛化
在已有的数据上，并未给出温度，气压等外部数据。这一部分，我们考虑将相关的外部参数引入。同样首先进行的是相关性的分析。
