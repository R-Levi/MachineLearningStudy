#非线性
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd


#高斯核
'''
Kg(x1,x2) = exp(-||x1-x2||^2/(2*sigma^2)
'''
def gaussian_kernel(x1,x2,sigma):
    sim = np.exp(-(np.sum((x1-x2)**2))/(2*sigma**2))
    return sim
#test
x1 = np.array([1.0,2.0,1.0])
x2 = np.array([0.0,4.0,-1.0])
#print(gaussian_kernel(x1,x2,sigma=2))

raw_data = sio.loadmat('ex6data2.mat')
data = pd.DataFrame(raw_data['X'],columns=['X1','X2'])
data['y'] = raw_data['y']

pos = data[data['y'].isin([1])]
neg = data[data['y'].isin([0])]

fig,ax = plt.subplots(figsize = (12,8))
ax.scatter(pos['X1'], pos['X2'], s=30, marker='x', label='Positive')
ax.scatter(neg['X1'], neg['X2'], s=30, marker='o', label='Negative')
ax.legend()
#plt.show()

#训练
'''
使用内置的RBF内核(径向基核函数)构建支持向量机分类器
RBF函数：exp(-gamma|u-v|^2)
'''
svc = svm.SVC(C=100,gamma=10,probability=True)
'''
SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=10, kernel='rbf',
  max_iter=-1, probability=True, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
'''
svc.fit(data[['X1','X2']],data['y'])
score = svc.score(data[['X1', 'X2']], data['y'])
print(score)

data['Probability'] = svc.predict_proba(data[['X1','X2']])[:,0]
'''
predict_proba返回的是一个n行k列的数组，
第i行第j列上的数值是模型预测第i个预测样本的标签为j的概率。
所以每一行的和应该等于1.
'''
#fig,ax = plt.subplots(figsize = (12,8))
#ax.scatter(data['X1'], data['X2'], s=30, c=data['Probability'],cmap='hot')
#plt.show()

#绘制边界
x1plot = np.linspace(np.min(data['X1']),np.max(data['X1']),100)
x2plot = np.linspace(np.min(data['X2']),np.max(data['X2']),100)
x1,x2 = np.meshgrid(x1plot,x2plot)#生成网格点矩阵
vals = np.zeros(np.shape(x1))
for i in range(np.size(x1,1)):
    this_x = np.vstack((x1[:,i],x2[:,i])).T#x1[:,i]一维的向量
    vals[:,i] = svc.predict(this_x)
c = plt.contour(x1, x2, vals,colors='b')
plt.savefig('非线性.png')
plt.show()