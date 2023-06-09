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

假设数据服从伯努利分布，通过极大似然估计，运用梯度下降求解，用于二分类问题，解决线性问题

本质上就是在线性回归后边加上了一个sigmoid函数

模型定义：

<img src="C:\Users\LEVI\AppData\Roaming\Typora\typora-user-images\image-20230330165728687.png" alt="image-20230330165728687" style="zoom:67%;" />

**这个e是怎么来的？**

利用贝叶斯建模，给定x，输出是C1类的概率为：

<img src="C:\Users\LEVI\AppData\Roaming\Typora\typora-user-images\image-20230330170452523.png" alt="image-20230330170452523" style="zoom:80%;" />

使用极大似然是估计来确定参数

<img src="C:\Users\LEVI\AppData\Roaming\Typora\typora-user-images\image-20230330170915124.png" alt="image-20230330170915124" style="zoom:67%;" />





## PCA

维度灾难

​		特征的维度越多，样本密度会变得更加稀疏，更容易找到一个超平面将样本区分，分错的概率越来越低。映射到高维，相当于在低维上学习了一个非线性的分类器。如果一直增加维度，样本越来越稀疏甚至趋于0，用很复杂的分类器去分类稀疏样本集，这样会造成过拟合，导致泛化效果不好，所以要降低维度，学一个泛化效果好的分类器，避免过拟合避免、维度灾难。



## **朴素贝叶斯**

贝叶斯分类是以贝叶斯定理为基础的一种分类算法，其主要思想为：先验概率+新的数据=后验概率
已知某条件概率，如何得到事件交换后的概率；即在已知P(B|A)的情况下求得P(A|B)。条件概率P(B|A)表示事件A已经发生的前提下，事件B发生的概率。其基本求解公式为：P(B|A)=P(AB)/P(A)

朴素贝叶斯假设**特征之间独立同分布**

步骤：

- 给定数据集：{xi，yi} xi是n维向量
- 根据给定数据集算出P（Y=yi）先验概率
- 对每个特征维度j，计算P（xj = xj|Y=yi）条件概率
- 根据测试输入的X* 对于每个类yi，都用贝叶斯公式计算，最大的就是X*所属的类

​                              		 计算P(X = x*|Y=yi)P(Y=yi)

如果有一个特征计算出来为0 或者连乘可能变得很小，可以优化改进

<img src="C:\Users\LEVI\AppData\Roaming\Typora\typora-user-images\image-20230331153902785.png" alt="image-20230331153902785" style="zoom:80%;" />

## 决策树

从信息论角度：信息熵越大，从而样本纯度越低。

<img src="C:\Users\LEVI\AppData\Roaming\Typora\typora-user-images\image-20230331220332591.png" alt="image-20230331220332591" style="zoom:67%;" />

ID3 算法的核心思想就是以信息增益来度量特征选择，选择信息增益最大的特征进行分裂。算法采用自顶向下的贪婪搜索遍历可能的决策树空间。其大致步骤为：

1. 初始化特征集合和数据集合；
2. 计算数据集合信息熵和所有特征的条件熵，选择信息增益最大的特征作为当前决策节点；
3. 更新数据集合和特征集合（删除上一步使用的特征，并按照特征值来划分不同分支的数据集合）；
4. 重复 2，3 两步，若子集值包含单一特征，则为分支叶子节点。

决策树构造准则：
$$
数据集D的信息熵：H(D) = -\sum_{k=1}^K\frac{|C_k|}{|D|}log_2\frac{|C_k|}{|D|} \\
C_k 集合D中属于k类样本的样本子集\\
条件熵，数据集D中A特征的条件熵：H(D|A) = \sum_{i=1}^{n}\frac{|D_i|}{|D|}H(D_i)\\
H(D_i) = -\sum_{k=1}^K\frac{D_{ik}}{Di}log_2\frac{D_{ik}}{D_i}\\
D_i 表示D中特征A中取第i个值的样本子集，D_{ik}表示D_i中属于第k类的样本子集
$$


- ID3最大信息熵增益  
  $$
  Gain(D,A) = H(D) - H(D|A)
  $$
  对于特征取值多的偏好，只能处理离散分布特征,没有剪枝，缺失值没有处理

- C4.5最大信息熵增益比：
  $$
  Gain_{ratio}(D,A) = \frac{Gain(D,A)}{H_A(D)}\\
  H_A(D) = -\sum_{i=1}^{n}\frac{|D_i|}{|D|}log_2\frac{|D_i|}{|D|}
  $$
  克服对于特征取值多的偏好，引入悲观剪枝策略进行后剪枝。将连续特征离散化，假设 n 个样本的连续特征 A 有 m 个取值，C4.5 将其排序并取相邻两样本值的平均数共 m-1 个划分点，分别计算以该划分点作为二元分类点时的信息增益，并选择信息增益最大的点作为该连续特征的二元离散分类点。

- CART－最大基尼指数

$$
Gini(D) = \sum_{k=1}^{K}\frac{C_k}{D}(1-\frac{C_k}{D}) = 1-\sum_{k=1}^{K}(\frac{C_k}{D})^2\\
Gini(D|A) = \sum_{i=1}^{n}\frac{|D_i|}{D}Gini(D_i)\\基尼指数代表了模型的不纯度，基尼系数越小，不纯度越低，特征越好。
$$

剪枝：

​	预剪枝：

​			控制树的深度

​			当前节点样本数量小于某个阈值，停止生长

​			计算每次分裂的测试集精确度，小于某个阈值，不在扩展

​	后剪枝：

​			基于代价复杂度的剪枝



## SVM





## 聚类

**聚类与分类的区别**
聚类是一种无监督学习，即数据不需要有标签即可。它与分类不同，分类是对有标签的数据进行的，是一种有监督学习。这是两者的区别。（举个例子，一堆人站在这里，没有标签，我们可以采用聚类来对这群人分组，如选取身高这个指标来对他们聚类。而如果是分类，比如男女分，按照每个人的性别标签即可。聚类不需要标签，只要我们自己选择一个指标，按指标来分簇即可。）
**聚类的概念**
聚类是按照某个指标（如样本之间的距离）把数据集分割成不同的类或者簇，使类内元素的相似性尽可能的大，类间元素的相似性尽可能小，通过这样来对相似的数据进行归簇，从而达到聚类的效果。
**聚类的步骤**
1.数据准备 ： 特征标准化（白化）
2.特征选择 ： 特征降维，选择最有效的特征
3.特征提取： 对选择的特征进行转换，提取出更有代表性的特征
4.聚类： 基于特定的度量函数进行相似度度量，使得同一类数据的相似度尽可能的贴近，不同类的数据尽可能分离，得到各个类的中心以及每个样本的类标签。
5.评估： 分析聚类结果，如距离误差和误差平方和（SSE）等
**常见的聚类算法**
    k-means、k-means++c、mean-shift、dbscan、层次聚类

## 集成学习
