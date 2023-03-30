#线性
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn import svm
'''
pd.DataFrame()
data：一组数据(ndarray、series, map, lists, dict 等类型)。
index：索引值，或者可以称为行标签。
columns：列标签，默认为 RangeIndex (0, 1, 2, …, n) 。
dtype：数据类型。
copy：拷贝数据，默认为 False。
'''
raw_data = sio.loadmat('ex6data1.mat')
data = pd.DataFrame(raw_data['X'],columns=['X1','X2'])
data['y'] = raw_data['y']
pos = data[data['y'].isin([1])]
neg = data[data['y'].isin([0])]
fig,ax = plt.subplots(figsize = (12,8))
ax.scatter(pos['X1'],pos['X2'],s=50,marker='x',label = 'POSSITIVE')
ax.scatter(neg['X1'],neg['X2'],s=50,marker='o',label = 'NEGATIVE')
ax.legend()
#plt.show()

svc = svm.SVC(C=100,kernel='linear')
svc.fit(data[['X1','X2']],data['y'])
score = svc.score(data[['X1', 'X2']], data['y'])
print(score)
theta = svc.coef_.flatten()#w0x0+w1x1+b=0
print(theta)
b = svc.intercept_
print(b)
x1 = np.linspace(np.min(data['X1']),np.max(data['X1']))
x2 =  -(theta[0]*x1+b)/theta[1]
plt.plot(x1,x2,'b-')
plt.title("C=100")
plt.savefig('Linear C=100.png')
plt.show()


