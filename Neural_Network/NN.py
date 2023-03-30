import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import scipy.io as sio
import math


input =400
hidden = 25
num = 10
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
Y = mainInfo['y'][:,0]#5000*1
m = np.size(Y)

rand_indices = np.random.permutation(m)#Randomly permute a sequence, or return a permuted range.
sel = X[rand_indices[0:100],:]
#show_digital(sel)

#加载参数
thetaInfo = sio.loadmat("ex3weights.mat")
theta1 = thetaInfo['Theta1']
theta2 = thetaInfo['Theta2']
#print(np.shape(theta1))(25, 401)
#print(np.shape(theta2))(10, 26)
nn_params = np.concatenate((theta1.flatten(),theta2.flatten()))#25*401+10*26的向量

#Feedforward
#sigmoid
def sigmoid(z):
    return 1/(1+np.exp(-z))
def sigmoidgradient(z):
    return sigmoid(z)*(1-sigmoid(z))
#损失函数
def nncost(params,input_layer_size,hidden_layer_size,num_labels,x,y,lam):
    theta1 = params[0:hidden_layer_size*(input_layer_size+1)].reshape(hidden_layer_size,input_layer_size+1)
    # 25 * 401
    theta2 = params[hidden_layer_size*(input_layer_size+1):].reshape(num_labels,hidden_layer_size+1)
    #10 * 26
    m = np.size(x,0)

    #前向传播
    a1 = np.concatenate((np.ones((m,1)),x),axis=1)#5000*401
    z2 = a1.dot(theta1.T)
    l2 = np.size(z2,0)
    a2 = np.concatenate((np.ones((l2,1)),sigmoid(z2)),axis=1)#5000*26
    z3 = a2.dot(theta2.T)
    a3 = sigmoid(z3)#5000*10
    yt = np.zeros((m,num_labels))#5000*10
    yt[np.arange(m),y-1] = 1#1表示属于该类，0不属于
    j = np.sum(-yt*np.log(a3)-(1-yt)*np.log(1-a3))/m
    #正则化
    reg_cost = np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2))
    j = j + 1 / (2 * m) * lam * reg_cost
    return j
#print(nncost(nn_params,input,hidden,num,X,Y,0))

#梯度函数
def nnGranfFun(params,input_layer_size,hidden_layer_size,num_labels,x,y,lam):
    theta1 = params[0:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, input_layer_size + 1)
    theta2 = params[hidden_layer_size * (input_layer_size + 1):].reshape(num_labels, hidden_layer_size + 1)
    m = np.size(x, 0)
    #25 * 401 + 10 * 26
    # 前向传播
    a1 = np.concatenate((np.ones((m, 1)), x), axis=1)  # 5000*401
    z2 = a1.dot(theta1.T)#5000*25
    l2 = np.size(z2, 0)
    a2 = np.concatenate((np.ones((l2, 1)), sigmoid(z2)), axis=1)  # 5000*26
    z3 = a2.dot(theta2.T)#5000*10
    a3 = sigmoid(z3)     #5000*10
    yt = np.zeros((m, num_labels))  # 5000*10
    yt[np.arange(m), y - 1] = 1
    #反向传播
    delta3 = a3-yt #5000*10
    delta2 = delta3.dot(theta2)*sigmoidgradient(np.concatenate((np.ones((l2, 1)), z2), axis=1))#5000*26
    theta2_grad = delta3.T.dot(a2)
    theta1_grad = delta2[:,1:].T.dot(a1)

    theta2_grad = theta2_grad / m
    theta2_grad[:, 1:] = theta2_grad[:, 1:] + lam / m * theta2[:, 1:]
    theta1_grad = theta1_grad / m
    theta1_grad[:, 1:] = theta1_grad[:, 1:] + lam / m * theta1[:, 1:]

    grad = np.concatenate((theta1_grad.flatten(),theta2_grad.flatten()))
    #print(np.shape(grad),np.shape(theta1_grad),np.shape(theta2_grad))
    #print(grad)
    return grad
#print(nnGranfFun(nn_params,input,hidden,num,X,Y,1).shape)

#随机初始化参数
def randInitTheta(lin,lout):
    epsilon_init = 0.12
    w = np.random.rand(lout,lin+1)*2*epsilon_init-epsilon_init
    return w
init_theta1 = randInitTheta(input, hidden)#initial_theta1为25x401
init_theta2 = randInitTheta(hidden, num)#initial_theta2为10x26
init_params = np.concatenate((init_theta1.flatten(),init_theta2.flatten()))
#print(init_theta1)
#print(init_theta2)

print("Training NN")
lamd = 1
param = op.fmin_cg(nncost,init_params,fprime=nnGranfFun,args=(input,hidden,num,X,Y,lamd),maxiter=250)
out_theta1 = param[0:(input+1)*hidden].reshape(hidden,input+1)
out_theta2 = param[(input+1)*hidden:].reshape(num,hidden+1)
print('OK')
#print(out_theta1)
#print(out_theta2)
#预测
def predict(theta1,theta2,x):
    m = np.size(x,0)
    x = np.concatenate((np.ones((m,1)),x),axis=1)
    a2 = sigmoid(x.dot(theta1.T))
    a2 = np.concatenate((np.ones((m,1)),a2),axis=1)
    a3 = sigmoid(a2.dot(theta2.T))
    p = np.argmax(a3,axis=1)+1
    return p
pre = predict(out_theta1,out_theta2,X)

print('准确率：{0}%'.format(np.sum(pre==Y)/np.size(Y,0)*100))