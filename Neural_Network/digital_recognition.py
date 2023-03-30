#神经网络实现手写数字识别
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.io as sio


mainInfo = sio.loadmat('ex3data1.mat')
X = mainInfo['X']
Y = mainInfo['y'][:,0]
m = np.size(Y)

rand_indices = np.random.permutation(m)#Randomly permute a sequence, or return a permuted range.
sel = X[rand_indices[0:100],:]

#显示数字
def display_digital(x,num):
    m,n = np.shape(x)
    width = round(math.sqrt(n))
    height = int(n/width)

    row = math.floor(math.sqrt(m))
    col = math.floor(math.sqrt(m))

    gap = 1
    back = -np.ones((gap+(height+gap)*row,gap+(width+gap)*col))
    cur = 0
    for i in range(row):
        for j in range(col):
            if cur>=m:
                break
            back[gap+i*(height+gap):gap+i*(height+gap)+height,gap+j*(width+gap):gap+j*(width+gap)+width]=x[cur,:].reshape(20,20,order="F")
            cur+=1
    plt.imshow(back)
    plt.axis('off')
    plt.title(num)
    plt.savefig("digital_show.png")
    plt.show()
#display_digital(sel)

#加载已经训练好的神经网络参数
weightInfo = sio.loadmat("ex3weights.mat")
theta1 = weightInfo['Theta1']
theta2= weightInfo['Theta2']
#print(theta1)
#print(theta2)


#sigmoid函数
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

#预测
def predict(theta1,theta2,x):
    m = np.size(x,0)
    x = np.concatenate((np.ones((m,1)),x),axis=1)
    temp1 = sigmoid(x.dot(theta1.T))
    temp = np.concatenate((np.ones((m,1)),temp1),axis=1)
    temp2 = sigmoid(temp.dot(theta2.T))
    p = np.argmax(temp2,axis=1)+1
    return p
pred = predict(theta1,theta2,X)
print("精确度：",np.sum(pred==Y)/np.size(Y))

#随机显示图像
d = np.random.permutation(m)
for i in range(10):
    pred = predict(theta1,theta2,X[d[i]:d[i]+1,:])
    if pred==10:
        pred=0;
    display_digital(X[d[i]:d[i]+1,:],pred)
    print(pred)