import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

'''初始化参数'''
#加载数据,初始化参数
data = np.loadtxt("ex2data2.txt",delimiter=',')
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

# 向高维扩展
def mapFeature(x1, x2):
    degree = 6
    col = int(degree*(degree+1)/2+degree+1) #28维
    out = np.ones((np.size(x1, 0), col))
    count = 1
    for i in range(1, degree+1):
        for j in range(i+1):
            out[:, count] = np.power(x1, i-j)*np.power(x2, j)
            count += 1
    return out

'''计算代价和梯度'''
X = mapFeature(x[:,0],x[:,1])
init_theta = np.zeros((np.size(X, 1),))
lamd = 1#正则化参数
#计算sigmoid函数
def sigmoid(z):
    a = 1.0/(1.0+np.exp(-z))
    return a
#计算损失
def costfunction(theta,x,y,lamd):
    m = len(y)
    h = sigmoid(x.dot(theta))
    #print(np.shape(x),np.shape(theta),np.shape(h),np.shape(y))
    #if np.sum(1 - h < 1e-10) != 0:  # 1-h < 1e-10相当于h > 0.99999999
    #    return np.inf  # np.inf 无穷大
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


cost = costfunction(init_theta,X,y,lamd)
grad = gradfunction(init_theta,X,y,lamd)
print("初始theta下的代价和梯度：",cost,grad)
#_=input('continue after pressing ENTER')

'''使用BFGS高级优化'''
'''调用scipy中的优化算法BFGS
    fun ：优化的目标函数
    x0 ：theta初值，一维数组，shape (n,)
    args ： 元组，可选，额外传递给优化函数的参数
    method：求解的算法，选择TNC则和fmin_tnc()类似
    jac：返回梯度向量的函数
'''
result = op.minimize(costfunction,x0=init_theta,method='BFGS',jac=gradfunction,args=(X,y,lamd))

theta = result.x
print('Cost at theta found by fmin_bfgs: ', result.fun) #result.fun为最小代价
print('theta: ', theta)
print('theta: ', result)
#绘制图像
def plotDescionBoundary(theta,x,y):
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    p1 = plt.scatter(x[pos, 1], x[pos, 2], marker='+', color='r')
    p2 = plt.scatter(x[neg, 1], x[neg, 2], marker='o', color='y')
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    z = np.zeros((np.size(u, 0), np.size(v, 0)))
    for i in range(np.size(u, 0)):
        for j in range(np.size(v, 0)):
            z[i, j] = mapFeature(np.array([u[i]]), np.array([v[j]])).dot(theta)
    z = z.T
    [um, vm] = np.meshgrid(u, v)
    plt.contour(um, vm, z, levels=[0])
    plt.legend((p1, p2), ('Admitted', 'Not admitted'), loc='upper right', fontsize=8)#加图例
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.title('lambda = 1')
    plt.savefig("ex2data2.png")
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