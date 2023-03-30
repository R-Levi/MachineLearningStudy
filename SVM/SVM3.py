import numpy as np
import scipy.io as sio
from sklearn import svm
import matplotlib.pyplot as plt

#第三个数据集，我们给出了训练和验证集，并且基于验证集性能为SVM模型找到最优超参数
raw_data = sio.loadmat('ex6data3.mat')

X = raw_data['X']
Xval = raw_data['Xval']
y = raw_data['y']
yval = raw_data['yval']

pos = np.where(y==1)
neg = np.where(y==0)
plt.scatter(X[pos,0],X[pos,1],marker='x')
plt.scatter(X[neg,0],X[neg,1],marker='o')
#plt.show()

#设置可选参数
C_val = [0.01,0.03,0.1,0.3,1,3,10,30,100]
gamma_val=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

#初始化变量
score = 0
params = {'C':None,'gamma':None}
final_svc = None
for C in C_val:
    for gamma in gamma_val:
        svc = svm.SVC(C=C,gamma=gamma)
        svc.fit(X,y)
        t_score = svc.score(Xval,yval)
        if t_score>score:
            score = t_score
            params['C']=C
            params['gamma']=gamma
            final_svc = svc
print(score)
print(params)

x1plot = np.linspace(np.min(Xval[:,0]),np.max(Xval[:,0]),100)
x2plot = np.linspace(np.min(Xval[:,1]),np.max(Xval[:,1]),100)
x1,x2 = np.meshgrid(x1plot,x2plot)
vals = np.zeros(np.shape(x1))
for i in range(np.size(x1,1)):
    this_x = np.vstack((x1[:,i],x2[:,i])).T
    vals[:,i]=final_svc.predict(this_x)
plt.contour(x1,x2,vals,colors='b')
plt.show()

