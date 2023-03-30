# MachineLearningStudy

机器学习记录，参考吴恩达系列

Scikit-learn(sklearn)是机器学习中常用的第三方模块，对常用的机器学习方法进行了封装，包括回归(Regression)、降维(Dimensionality Reduction)、分类(Classfication)、聚类(Clustering)等方法

## **线性回归**

给定数据 	D={x1,y1;x2,y2;x3,y3;...}  用于回归问题

假设函数：
$$
f(x) = w^Tx+b
$$
损失函数：
$$
L(w) = \sum_{i=1}^N||w^Tx_i+b-y_i||_2^2
$$
梯度下降或者最小二乘（SVD）求解

假设误差服从高斯分布，最小化MSE损失和极大似然估计是一致的

## 逻辑回归

假设数据服从伯努利分布，通过极大似然估计，运用梯度下降求解，用于二分类问题

模型定义：

<img src="C:\Users\LEVI\AppData\Roaming\Typora\typora-user-images\image-20230330165728687.png" alt="image-20230330165728687" style="zoom:67%;" />
