import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import scipy.io as sio
import math
from sklearn.preprocessing import OneHotEncoder

#可视化数据
def show_digital(x):
    m,n = np.shape(x)
    width = round(math.sqrt(n))
    height = int(n / width)
    row = math.floor(math.sqrt(m))
    col = math.floor(math.sqrt(m))
    gap = 1
    back = -np.ones((gap+(height+gap)*row,gap+(width+gap)*col))

    cur = 0
    for i in range(row):
        for j in range(col):
            if cur>=m:
                break
            back[gap+i*(height+gap):gap + i*(height+gap)+height,gap + j*(width+gap):gap+j*(width+gap)+width]=x[cur,:].reshape(20, 20, order="F")
            cur+=1
    plt.imshow(back)
    plt.axis('off')
    plt.show()

#加载数据
mainInfo = sio.loadmat('ex3data1.mat')
X = mainInfo['X']#5000*400
y = mainInfo['y']#5000*1
m = np.size(X,0)
#print(m)
#我们也需要对我们的y标签进行一次one-hot 编码。
# one-hot 编码将类标签n（k类）转换为长度为k的向量，
# 其中索引n为“hot”（1），而其余为0。
# Scikitlearn有一个内置的实用程序，我们可以使用这个。
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y)

#初始化设置
input_size = 400
hidden_size = 25
num_lables = 10
lamda = 1
#随机初始化
params = (np.random.random(size=hidden_size*(input_size+1)+num_lables*(hidden_size+1))-0.5)*0.25#10258
theta1 = params[0:hidden_size*(input_size+1)].reshape(hidden_size,input_size+1)#25*401
theta2 = params[hidden_size*(input_size+1):].reshape(num_lables,hidden_size+1)#10*26


#sigmoid
def sigmoid(z):
    return 1/(1+np.exp(-z))
def forward_propatage(X,theta1,theta2):
    m = np.size(X,0)
    a1 = np.concatenate((np.ones((m,1)),X),axis=1)
    z2 = a1.dot(theta1.T)
    a2 = np.concatenate((np.ones((m,1)),sigmoid(z2)),axis=1)
    z3 = a2.dot(theta2.T)
    h = sigmoid(z3)
    return a1,z2,a2,z3,h
a1,z2,a2,z3,h = forward_propatage(X,theta1,theta2)
#print(a1.shape,z2.shape,a2.shape,z3.shape,h.shape)
#(5000, 401) (5000, 25) (5000, 26) (5000, 10) (5000, 10)
def nn_cost(params,input_size,hidden_size, num_lables, X, y, lamda):
    m = X.shape[0]
    theta1 = params[0:hidden_size * (input_size + 1)].reshape(hidden_size, input_size + 1)  # 25*401
    theta2 = params[hidden_size * (input_size + 1):].reshape(num_lables, hidden_size + 1)  # 10*26
    a1, z2, a2, z3, h = forward_propatage(X, theta1, theta2)
    J = np.sum(-y*np.log(h)-(1-y)*np.log(1-h))/m
    #正则化
    reg = np.sum(np.power(theta1[:,1:],2))+np.sum(np.power(theta2[:,1:],2))
    reg = reg*lamda/(2*m)
    J = J+reg
    return J
J = nn_cost(params,input_size,hidden_size,num_lables,X,y_onehot,lamda)


#反向传播
def sigmoid_gradient(z):
    return np.multiply(sigmoid(z),1-sigmoid(z))


def backprop(params, input_size, hidden_size, num_labels, X, y, lamda):
    J = nn_cost(params, input_size, hidden_size, num_lables, X, y, lamda)
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))#25*401
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))#10*26
    a1, z2, a2, z3, h = forward_propatage(X, theta1, theta2)
    l2 = np.size(z2, 0)
    d3 = h-y #5000*10
    d2 = np.multiply(d3.dot(theta2),sigmoid_gradient(np.concatenate((np.ones((l2, 1)), z2), axis=1)))#5000*26
    delta2 = d3.T.dot(a2)
    delta1 = d2[:, 1:].T.dot(a1)
    '''
    delta1 = np.zeros(theta1.shape)  # (25, 401)
    delta2 = np.zeros(theta2.shape)  # (10, 26)
    for t in range(m):  # 遍历每个样本
        a1t = a1[t, :]  # (1, 401)
        z2t = z2[t, :]  # (1, 25)
        a2t = a2[t, :]  # (1, 26)
        ht = h[t, :]  # (1, 10)
        yt = y[t, :]  # (1, 10)
        d3t = ht - yt#1*10
        z2t = np.insert(z2t, 0, values=np.ones(1))  # (1, 26)
        d2t = np.multiply((d3t * theta2), sigmoid_gradient(z2t))  # (1, 26)
        delta1 = delta1 + (d2t[:, 1:]).T * a1t#26*401
        delta2 = delta2 + d3t.T * a2t#10*26
    '''

    # STEP8：加入正则化
    # your code here  (appro ~ 1 lines)
    delta1[:, 1:] = delta1[:, 1:] + (theta1[:, 1:] * lamda) / m
    delta2[:, 1:] = delta2[:, 1:] + (theta2[:, 1:] * lamda) / m
    # STEP9：将梯度矩阵转换为单个数组
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))
    return J,grad
J,grad = backprop(params, input_size, hidden_size, num_lables, X, y_onehot, lamda)
#print(J,grad.shape)
'''
#截断牛顿法
fmin = op.minimize(fun=backprop,x0=params,args=(input_size, hidden_size, num_lables, X, y_onehot, lamda),
                   method='TNC',jac=True,options={'maxiter':250})
print(fmin)
sio.savemat("out_params",{'params':fmin.x})
'''


paramsData = sio.loadmat("out_params.mat")
out_params = paramsData['params'][0]
theta1 = out_params[:hidden_size * (input_size + 1)].reshape(hidden_size,input_size+1)
theta2 =out_params[hidden_size * (input_size + 1):].reshape(num_lables,hidden_size+1)
a1, z2, a2, z3, h = forward_propatage(X, theta1, theta2)
y_pred = np.array(np.argmax(h, axis=1) + 1)
#print(y_pred)
correct = [1 if a==b else 0 for (a,b) in zip(y_pred,y)]
print("accuracy = {0}%".format(sum(map(int,correct))/len(y)*100))