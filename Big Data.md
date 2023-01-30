# Big Data

[toc]

## 物理海洋学

描述海洋的运动

目标：

1. 理解时间和空间上的过程
2. 模拟这些过程
3. 预测

盐度、温度和风都会导致洋流的产生

海洋表面温度图-- 温度导数？ -->空气海洋热量交换图

研究方法：

1. 理论模型：非线性、乱流
2. 观察：时间与空间
3. 数学模型：对现实状况的模拟，只包含已知的物理学知识

不同的数学模型：

1. forced：人工喂养数据，由过去的数据推测未来的数据
2. coupled：不只是海洋模型，是一个气候模型，包括海冰、海波、大气、陆冰等模型，runs at the same time as the ocean model

一般来说，coupled 模型的结果比 forced 模型要差，他们会被同时用于预测长期的数据

## 应用于物理海洋学的数据科学

方法：观察、反思、实验

主要任务：预测和描述

常用预测模型：SVM、多变量线性回归、人工智能神经网络（ANN）

## 云平台

什么是云计算：通过远程的平台去使用计算或存储资源

五个特征：

1. 广泛的网络接口
2. 

三种服务模式：

1. Software as a service（SaaS）
2. Platform as a service（PaaS）
3. Infrastructure as a service（IaaS）

https://zhuanlan.zhihu.com/p/28532380

http://c.biancheng.net/view/3787.html

四种使用模式：

1. Public
2. Private
3. Hybrid
4. Community

主要优点：

1. 更低的计算消耗
2. 性能优化
3. 减少软件消耗
4. 即时的软件更新
5. 没有限制的存储能力

## Practice on Pangeo

Geoscience dataset are N-Dimensional arrays of variables (eg: temperature)

N is up to 5 (3 space + 1 time + 1 ensemble)

### Xarray

N 维的 Panda

包含三类数据：Dimension，Variable，Attribute

### Dask

Python 中用于并行计算的库，可以加快计算速度

Xarray 会自动使用 Dask

### Zarr

文件管理包，能够把大型的数据包分割为多个小的数据包，并把它们存储在云平台上的分布式系统上。

懒惰管理，每次只上传需要的数据

Zarr + Dask + Xarray = **A flexible framework for parallel computing on large labelled arrays**

## 大数据的价值

分布式系统的目标：连接不同电脑，从而实现计算资源和存储资源的共享

Vs of BigData：

1. Volume：数据量
2. Variety：数据形式各异
3. Velocity：数据流分析
4. Veracity：数据的不确定性

### Hadoop：一个用于简化建造分布式系统的框架

Hadoop 的管理原则：

一个 Hadoop 的 cluster 应该能够组织成百上千个节点，每个节点能够提供存储和计算资源

一个 Hadoop 的 cluster 应该能够在可接受的代价和延迟下，存储并处理大量的数据

如果一个节点出现故障，不能导致计算的停止或数据的丢失

### HDFS：Hadoop Distributed File System

一个文件管理系统（SGF），用来在 cluster 上读写数据

每个 bloc 会被保存至少 3 次，用来保证可用性和安全性

一种特殊的 SGF 只能被写一次（RMWO，多读一写），一旦被写，数据就不可更改

（RMWM 多读多写）的 SGF 可以被重复更改，每次更改，被复制的部分会被链接更改

### MapReduce

https://juejin.cn/post/6844903687094009863

A data processing methodology made popular by Hadoop
It describes a way that multiple computational units can work together to process a large scale dataset whilst acting independently and not depending on one another

高计算能力，高用户透明性，模型相对简单

## Predictive statistics

Predictive statistics : Given existing data, find the parameters for a chosen model that let us treat new data

Descriptive statistics : Given existing data, find the parameters for a chosen model so that we get the best possible description of the existing data

Exploratory data analysis : Given existing data, aim to discover underlying patterns so that we can form hypotheses and develop statistical models of the data

海表面温度越高，海面越高

盐度？

接下来讲到了：

1. 线性回归
   1. 由 x 拟合 y 和由 y 拟合 x 得到的结果并不一定恰好相反
   2. 用线性回归可以模拟非线性关系（e.g. x=cos(t), y=2*cos(t)）
   3. [Python 代码](https://github.com/obidam/ds2-2023/blob/main/practice/predictive_statistics/1_Simple_linear_regression.ipynb)
2. 多变量线性回归
   1. 变量增多，因此使用矩阵形式
   2. [Python 代码](https://github.com/obidam/ds2-2023/blob/main/practice/predictive_statistics/2_Multiple_linear_regression.ipynb)
3. 非线性回归
   1. 支持向量回归（SVR）， $\epsilon$ 表示接受区间，epsilon controls how much data surrounds the fitted line at each point
   2. [Python 代码](https://github.com/obidam/ds2-2023/blob/main/practice/predictive_statistics/3_Nonlinear_regression_with_Support_Vector_Machines.ipynb)
4. 分类
   1. 逻辑回归
   2. [Python 代码]([https://github.com/obidam/ds2-2023/blob/main/practice/predictive_statistics/4_Logistic_regression.ipynb])

## Explanatory statistics

数据集的模式 **Paterns**  ：

1. correlations：相关性
2. trends：趋势
3. clusters：集群
4. trajectories：轨迹
5. anomalies：偏差

**主要的描述方法 - 相关性分析（Association Analysis）** 

寻找事物间的相关性（买尿布的同时会买啤酒，气候性火灾和极端降雨情况的联系）

**主要的描述方法 - 集群（Clustering）** 

把数据分组，寻找组内的相关性和组间的差异性

集群分类的不同模式：

1. Hierarchical and Partitional Clustering
   1. Hierarchical：set of nested clusters organised as a tree
   2. Partitional：otherwise
2. Hard and Fuzzy clustering
   1. Hard：Exclusive VS Overlapping
      1. Exclusive：one data cannot be simultaneously in more than 1 cluster
      2. Overlapping：it can
   2. Fuzzy：express probability for one data to be in each clusters
3. Complete and Partial clustering
   1. Complete：all data are classified
   2. Partial：not necessarily (outliers…)

分类方法：

1. Partitioning-based
2. Hierarchical-based
3. Density-based
4. Grid-based
5. Model-based

**One key issue in clustering:** The number of cluster is often an input parameter of the method

集群太多 -> Overfit

集群太少 -> Underfit

For model-based methods, one can use **Bayesian Information Criteria (BIC)** to determine the number of clusters（PPT-178）

为了 BIC 能更好地工作，应该让数据集仅包含有用信息，多余的信息会使算法困惑，采样要尽可能独立

**Clustering method** ：

1. K-means：大多数方法不能用于大量数据集，但 K-means 可以，因为计算距离很快
2. GMM：高斯混合模型，

**主要的描述方法 - 误差检测（Anomaly Detection）** 

An anomaly is an *unusual* object, or is *inconsistent* with other objects

Approaches fall into 3 categories：

1. model-based technics：the bunch obeys a model, an anomaly does not fit the model well, an outlier is a data that has a low probability with respect to the PDF model of the data
2. proximity-based technics：the bunch is grouped, an anomaly is distant from the group, an outlier is a data that is in a region of low density, which can be defined as the inverse distance of the k-nearest neighbours
3. density-based technics：the bunch is dense, an anomaly is located in low density region, an anomaly will have a poor probability to belong or will be at a large distance to its labelled cluster
