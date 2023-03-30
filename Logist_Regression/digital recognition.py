#手写数字识别
import numpy as np
import matplotlib.pyplot as plt
import math
#用于打开matlab文件
import scipy.io as sio
import scipy.optimize as op



#读取matlab的数据集，包括是20*20的像素以及0-9标签
matinfo = sio.loadmat("ex3data1.mat")
X = matinfo['X']#20*20像素
Y = matinfo['y'][:,0]#标签
#print(Y)
#y = 1*(Y==10)
#print(y)
#print(np.shape(X),np.shape(Y))
m = np.size(X,0)
rand_indices = np.random.permutation(m)#打乱顺序
dig = X[rand_indices[0:100],:]#选择100行


#显示数字
def show_num(x):
    m,n = np.shape(x)
    width = round(math.sqrt(np.size(x,1)))#单张图片宽度
    height = int(n/width)#高度
    row = math.floor(math.sqrt(m))#一行几个图
    col = math.ceil(m/row)#一列几个图
    gap = 1 #每个图间隔1
    background = -np.ones((gap+row*(height+gap),gap+col*(width+gap)))#初始化图层
    cur = 0
    for i in range(row):
        for j in range(col):
            if cur>=m:
                break
            background[gap+i*(height+gap):gap+i*(height+gap)+height,gap+j*(width+gap):gap+j*(width+gap)+width]=x[cur,:].reshape(20,20,order='F')
            #  order=F指定以列优先，在matlab中是这样的，python中需要指定，默认以行
            cur+=1
    plt.imshow(background,cmap='gray')
    plt.axis('off')
    #plt.savefig('digital_show.png')
    plt.show()
#show_num(dig)



#逻辑回归多分类
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
def costfunction(theta,x,y,lamd):
    m = len(y)
    h = sigmoid(x.dot(theta))
    j = -1/m*(y.dot(np.log(h))+(1-y).dot(np.log(1-h)))+lamd/(2*m)*theta[1:].dot(theta[1:])
    #print(j)
    return j
#计算梯度
def gradfunction(theta,x,y,lamd):
    m = len(y)
    h = sigmoid(x.dot(theta))
    grad = np.zeros(np.size(theta,0))
    grad[0] = 1/m*(x[:,0].dot(h-y))
    grad[1:] = 1/m*(x[:,1:].T.dot(h-y))+lamd*theta[1:]/m
    return grad
def OnevsAll(x,y,lamd):
    m,n = np.shape(x)
    all_theta = np.zeros((10,n+1))#一个类一组Θ
    x = np.concatenate((np.ones((m,1)),x),axis=1)
    for i in range(10):
        num = 10 if i==0 else i
        # 拟牛顿法：BFGS算法
        init_theta = np.zeros((n+1,))#401维的向量
        result = op.minimize(costfunction,x0=init_theta,method='BFGS',jac=gradfunction,args=(x,y==num,lamd))
        all_theta[i,:]=result.x
    print(result.success)
    return all_theta

lamd=0.1
onevs_all = OnevsAll(X, Y,lamd)
# 预测值函数
def predictOneVsAll(all_theta, x):
    m = np.size(x,0)
    x = np.concatenate((np.ones((m, 1)), x), axis=1)# 这里的axis=1表示按照列进行合并
    p = np.argmax(x.dot(all_theta.T), axis=1)#np.argmax(a)取出a中元素最大值所对应的索引（索引值默认从0开始）,axis=1即按照行方向搜索最大值
    return p

pred = predictOneVsAll(onevs_all, X)
print('Training Set Accuracy: ', np.sum(pred == (Y % 10))/np.size(Y, 0))


'''
#使用sklearn预测
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X,Y)
pred1 = model.predict(X)
print(u"预测准确度为：%f%%"%np.mean(np.float64(pred1 == Y)*100))
'''


