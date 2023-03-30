#梯度上升算法
import numpy as np
import matplotlib.pyplot as plt
#加载数据
def loadDataSet():
    datamat = []
    lablemat= []
    fr = open('ex2data1.txt')
    for line in fr.readlines():
        lineArr = line.strip().split(',')
        datamat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        lablemat.append([int(lineArr[2])])
    return datamat,lablemat
def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def gradAscent(dataIn,lableIn):
    dataMatrix = np.mat(dataIn)
    lableMatrix = np.mat(lableIn)
    m,n = np.shape(dataMatrix)
    alpha = 0.001
    cycles = 300000
    w = np.ones((n,1))
    for k in range(cycles):
        h = sigmoid(dataMatrix.dot(w))
        error = (lableMatrix-h)
        w = w+alpha*dataMatrix.transpose().dot(error)
    return w

def plotFit(wei):
    #矩阵变数组
    w = wei.getA()
    dataMat, lableMat = loadDataSet()
    dataArr = np.asarray(dataMat)
    lableArr = np.asarray(lableMat)
    n = np.shape(dataMat)[0]
    x1=[];y1=[];x2=[];y2=[]
    for i in range(n):
        if lableArr[i]==1:
            x1.append(dataArr[i,1])
            y1.append(dataArr[i,2])
        else:
            x2.append(dataArr[i, 1])
            y2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x1,y1,c='red',marker='s')
    ax.scatter(x2,y2,c='green')
    x = np.arange(30, 90, 1)
    y = (-w[0]-w[1]*x)/w[2]
    plt.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
if __name__ == '__main__':
    dadaset,lableset = loadDataSet()
    wei = gradAscent(dadaset, lableset)
    print(wei)
    plotFit(wei)
