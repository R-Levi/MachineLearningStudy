import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

'''初始化参数'''
#加载数据,初始化参数
data = np.loadtxt("ex2data1.txt",delimiter=',')
x = data[:,0:2]
y = data[:,2]

#绘制散点图
def drawdata(x,y):
    plt.scatter(x[:,0],x[:,1],c = np.squeeze(y),cmap='coolwarm')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("exam")
    plt.show()
#drawdata(x,y)
#_=input('continue after pressing ENTER')


'''计算代价和梯度'''
m,n = np.shape(x)#数据总数和特征个数
X = np.concatenate((np.ones([m,1]),x),axis=1)
# 这里的axis=1表示按照列进行合并(axis=0表示按照行进行合并)
#print(X)
init_theta = np.zeros([n+1,1])
#计算sigmoid函数
def sigmoid(z):
    a = 1.0/(1.0+np.exp(-z))
    return a
#计算损失
def costfunction(theta,x,y):
    m = len(y)
    h = sigmoid(x.dot(theta))
    if np.sum(1 - h < 1e-10) != 0:  # 1-h < 1e-10相当于h > 0.99999999
        return np.inf  # np.inf 无穷大
    j = -1/m*(y.dot(np.log(h))+(1-y).dot(np.log(1-h)))
    return j
#计算梯度
def gradfunction(theta,x,y):
    m = len(y)
    grad = 1/m*(x.T.dot(sigmoid(x.dot(theta))-y))
    return grad


cost = costfunction(init_theta,X,y)
grad = gradfunction(init_theta,X,y)
#print("初始theta下的代价和梯度：",cost,grad)
#_=input('continue after pressing ENTER')

'''使用BFGS高级优化'''
'''调用scipy中的优化算法BFGS
    fun ：优化的目标函数
    x0 ：theta初值，一维数组，shape (n,)
    args ： 元组，可选，额外传递给优化函数的参数
    method：求解的算法，选择TNC则和fmin_tnc()类似
    jac：返回梯度向量的函数
'''
result = op.minimize(costfunction,x0=init_theta,method='BFGS',jac=gradfunction,args=(X,y))

theta = result.x
print('Cost at theta found by fmin_bfgs: ', result.fun) #result.fun为最小代价
print('theta: ', theta)
#绘制图像
def plotDescionBoundary(theta,x,y):
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    p1 = plt.scatter(x[pos, 1], x[pos, 2], marker='+', color='r')
    p2 = plt.scatter(x[neg, 1], x[neg, 2], marker='o', color='y')
    plot_x = np.array([np.min(x[:, 1]) - 2, np.max(x[:, 1] + 2)])
    plot_y = -1 / theta[2] * (theta[1] * plot_x + theta[0])
    plt.plot(plot_x, plot_y,linewidth = 5)
    plt.legend((p1, p2), ('Admitted', 'Not admitted'), loc='upper right', fontsize=8)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig('ex2data1.png')
    plt.show()

plotDescionBoundary(theta,X,y)

'''预测'''
# 预测给定值
def predict(theta, x):
    m = np.shape(x)[0]
    p = np.zeros((m,))
    pos = np.where(x.dot(theta) >= 0)
    neg = np.where(x.dot(theta) < 0)
    p[pos] = 1
    p[neg] = 0
    return p
p = predict(theta, X)
print('Train Accuracy: ', np.sum(p == y)/len(y))