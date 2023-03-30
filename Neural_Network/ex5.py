#underfit overfit
import numpy as np
import scipy.io as sio
import scipy.optimize as op
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def load_data():
    d = sio.loadmat('ex5data1.mat')
    #return np.ravel(d['X']),d['y'],d['Xval'], d['yval'], d['Xtest'], d['ytest']
    return map(np.ravel,[d['X'],d['y'],d['Xval'], d['yval'], d['Xtest'], d['ytest']])
X, y, Xval, yval, Xtest, ytest = load_data()#训练集 cv集 测试机
#print(X,y, Xval, yval, Xtest, ytest)
'''
df = pd.DataFrame({'water_level':X,'flow':y})
sns.lmplot('water_level','flow',data=df,fit_reg=False,size=5)
plt.show()
'''


X,Xval,Xtest = [np.insert(x.reshape(x.shape[0],1),0,np.ones((x.shape[0])),axis=1) for x in (X,Xval,Xtest)]

def cost(theta,X,y):
    m = X.shape[0]
    inner = np.dot(X,theta) -y
    sum = np.dot(inner.T,inner)
    cost = sum/(2*m)
    return cost
theta = np.ones(X.shape[1])
c = cost(theta,X,y)
#print(c)
def gradient(theta,X,y):
    m = X.shape[0]
    grad = (np.dot(X.T,np.dot(X,theta)-y))/m
    return grad
g = gradient(theta,X,y)
#print(g)

def reg_cost(theta,X,y,l=1):
    m = X.shape[0]
    reg_term  = (1/(2*m))*np.power(theta[1:],2).sum()
    return cost(theta,X,y)+reg_term
def reg_gradient(theta,X,y,l=1):
    m = X.shape[0]
    reg_term = theta.copy()
    reg_term[0] = 0
    reg_term = (1/m)*reg_term
    return gradient(theta,X,y)+reg_term
#print(reg_gradient(theta,X,y))

#拟合数据
def linear_regression_np(X,y,l=1):
    theta = np.ones(X.shape[1])
    res = op.minimize(fun=reg_cost,x0=theta,args=(X,y,l),method='TNC',jac=reg_gradient,options={'disp':True})
    return res

theta = np.ones(X.shape[0])
final_theta = linear_regression_np(X, y, l=0).get('x')
#print(final_theta)
b = final_theta[0]
m = final_theta[1]
'''
plt.scatter(X[:,1],y)
plt.plot(X[:,1],X[:,1]*m+b)
plt.legend(["predicted data","Training data"])
plt.show()
'''
'''
1.使用训练集的子集来拟合应模型
2.在计算训练代价和交叉验证代价时，没有用正则化
3.记住使用相同的训练集子集来计算训练代价
TIP：向数组里添加新元素可使用append函数
'''
trainint_cost,cv_cost=[],[]
m = X.shape[0]
for i in range(1,m+1):
    res = linear_regression_np(X[:i, :], y[:i], l=0)
    tc = reg_cost(res.x,X[:i,:],y[:i],l=0)
    cv = reg_cost(res.x,Xval,yval,l=0)
    trainint_cost.append(tc)
    cv_cost.append(cv)
plt.plot(np.arange(1,m+1),trainint_cost,label='training cost')
plt.plot(np.arange(1,m+1),cv_cost,label='cv cost')
plt.legend(loc = 2)
#plt.show()
#创建多项式特征
def prepare_poly_data(*args,power):
    def perpare(x):
        #特征映射
        df = poly_features(x,power=power)
        #归一化
        ndarr = normalize_feature(df).as_matrix()

        return np.insert(ndarr,0,np.ones(ndarr.shape[0]),axis=1)
def poly_features(x,power,as_ndarry = False):#特征映射
    data = {'f{}'.format(i):np.power(x,i) for i in range(1,power+1)}
    df = pd.DataFrame(data)
    return df.as_matrix() if as_ndarry else df

def normalize_feature(df):
    return df.apply(lambda column:(column-column.mean())/column.std())