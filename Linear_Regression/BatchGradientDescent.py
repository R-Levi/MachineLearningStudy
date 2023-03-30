import numpy as np
import os
import matplotlib.pyplot as plt
#导入warnings包，利用过滤器来实现忽略警告语句。
import warnings
warnings.filterwarnings(action="ignore",message="^internal gelsd")
np.random.seed(1)
#保存图片
PROJECT_ROOT_DIR="."
MODEL_ID="linear_models"
def save_pic(pic_id):
    path = os.path.join(PROJECT_ROOT_DIR,"image",MODEL_ID,pic_id+".png")
    print("SAVE PICTURE",pic_id)
    plt.savefig(path,format="png")

#绘图
def draw_pic(X,Y,eta,res,num):
    plt.subplot(num)
    plt.scatter(X,Y,c ="blue",s=5)#画散点图
    plt.ylabel("Y",rotation=0)
    plt.xlabel("X")
    plt.title("α={}".format(eta))
    predict_x = np.arange(0, 3)
    predict_y = res[0] + res[1] * predict_x
    plt.plot(predict_x, predict_y,"k-",linewidth=2)


#BatchGradinetDescent
#计算代价函数
def cost_function(x,y,theta,m):
    cost = np.sum((x.dot(theta)-y)**2)
    return cost/(2*m)
#计算梯度
def gradient(x,y,theta):
    grad = np.empty(len(theta))
    grad[0] = np.sum(x.dot(theta)-y)
    r_x = x[:,1]
    for i in range(1,len(theta)):
        grad[i] = np.transpose(x.dot(theta) - y).dot(r_x)
    return grad.reshape([2,1])
#梯度下降
def gradient_descent(x,y,theta,a,m):
    while True:
        old_theta = theta
        grad  = gradient(x,y,theta)
        theta = theta - a * grad
        if abs(cost_function(x,y,old_theta,m)-cost_function(x,y,theta,m)) < 1e-15:
            break
    return theta

if __name__ == '__main__':
    #创建数据
    np.random.seed(0)
    X = 2*np.random.rand(200,1)#100行数据
    Y = 4+3*X+np.random.rand(200,1)
    m=200
    #plt.scatter(X,Y,c ="black",s=10)
    # X转化为矩阵方便后面的的计算 ones生成全1的数组，c_按行拼接
    new_X = np.c_[np.ones((m, 1)), X]
    theta = np.random.randn(2,1)

    #绘图结果
    plt.figure(figsize=(10,5))
    res = gradient_descent(new_X,Y,theta,0.0001,m)
    draw_pic(X,Y,0.0001,res,131)
    res = gradient_descent(new_X, Y, theta, 0.0005, m)
    draw_pic(X, Y, 0.0005, res, 132)
    res = gradient_descent(new_X, Y, theta, 0.002, m)
    draw_pic(X, Y, 0.002, res, 133)
    save_pic("batchGradientDescent_result")
    plt.show()
