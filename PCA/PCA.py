#主成分分析（principal component analysis）
#降维
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

data = sio.loadmat('ex7data1.mat')
X = data['X']
'''
fig,ax = plt.subplots(figsize=(12,8))
ax.scatter(X[:, 0], X[:, 1])
plt.show()
'''

'''
PCA的算法相当简单。 在确保数据被归一化之后，输出仅仅是原始数据的协方差矩阵的奇异值分解。 
Tip：矩阵奇异值分解可以使用np.linalg.svd(X)函数，其中X是待分解矩阵。
'''

def pca(X):
    #归一化
    X = (X-np.mean(X))/np.std(X)

    X = np.matrix(X)
    cov = (X.T*X)/X.shape[0]

    #奇异值分解
    U,S,V = np.linalg.svd(cov)
    return U,S,V

#低维映射，我们将实现一个计算投影并且仅选择顶部K个分量的函数，有效地减少了维数。
def project_data(X,U,k):
    U_reduced = U[:,:k]
    return np.dot(X,U_reduced)

#反向转换
def recover_data(Z,U,k):
    U_reduced = U[:,:k]
    return np.dot(Z,U_reduced.T)
#test
U,S,V = pca(X)
print(U)
print(S)
print(V)
Z = project_data(X,U,1)
#print(Z)
x_recovered = recover_data(Z,U,1)
#print(x_recovered)
plt.plot(Z[:,0])
plt.show()
fig,ax = plt.subplots(figsize = (12,8))
ax.scatter(list(x_recovered[:,0]),list(x_recovered[:,1]))
plt.show()
