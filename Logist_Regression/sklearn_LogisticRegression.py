# -*- coding: utf-8 -*-
#sklearn实现逻辑回归
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

#加载数据
def loaddata(filename,split,dataType):
    return np.loadtxt(filename,delimiter=split,dtype=dataType)
'''
data = loaddata("iris.data",",",np.str)
for i in range(0,data.shape[0]):
    if data[i,4]=='Iris-setosa':
        data[i, 4]='0'
    if data[i, 4] == 'Iris-versicolor':
        data[i, 4] = '1'
    if data[i, 4] == 'Iris-virginica':
        data[i, 4] = '2'
np.savetxt("out.data",data,fmt="%s",delimiter=',')
'''
def logiststicRegression():
    data = loaddata("out.data",",",np.float)
    #print(data)
    x = data[:,:2]
    y = data[:,-1:]
    #划分训练集
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
    # 逻辑回归
    model = LogisticRegression()
    model.fit(x_train, y_train)

    #画图
    #print(model.coef_)#特征系数
    #print(model.intercept_)#截距
    x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5  # 第1维度网格数据预备
    y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5  # 第2维度网格数据预备
    h=0.01
    xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))#创建网格
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)  # 将 Z 矩阵转换为与 xx 相同的形状
    plt.figure(figsize=(5,5))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)  # 作网格图
    col = ['c', 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    plt.scatter(x_test[:, 0], x_test[:, 1], c=np.squeeze(y_test), edgecolors='k', cmap='coolwarm')  # 画出预测的结果
    plt.xlabel('Sepal length')  # 作x轴标签
    plt.ylabel('Sepal width')  # 作y轴标签
    plt.xlim(xx.min(), xx.max())  # 设置x轴范围
    plt.ylim(yy.min(), yy.max())  # 设置y轴范围
    plt.savefig("sk_out.png")
    plt.show()

    '''
        #归一化
        scaler = StandardScaler()
        # scaler.fit(x_train)
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.fit_transform(x_test)
    '''
    #预测
    predict = model.predict(x_test)
    predict = np.hstack((predict.reshape(-1, 1), y_test.reshape(-1, 1)))
    print(model.score(x_test,y_test))
if __name__ == '__main__':
    logiststicRegression()