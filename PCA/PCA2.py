#PCA用于脸部图像
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


face = sio.loadmat('ex7faces.mat')
X = face['X']
#print(X.shape)(5000, 1024)
'''
face = np.reshape(X[50,:],(32,32))
plt.imshow(face)
plt.show()
'''

#展示图像
def draw_image(X,n):
    pic_size = int(np.sqrt(X.shape[1]))
    grid_size = int(np.sqrt(n))

    xn = X[:n,:]
    fig,ax = plt.subplots(nrows=grid_size,ncols=grid_size,
                          sharex=True,sharey=True,figsize = (8,8))
    for r in range(grid_size):
        for c in range(grid_size):
            ax[r,c].imshow((xn[grid_size*r+c,:]).reshape((pic_size,pic_size)))
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))

#draw_image(X,100)
#plt.show()
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

face = np.reshape(X[50,:],(32,32))
plt.imshow(face)
plt.show()

#在面数据集上运行PCA，并取得前100个主要特征。
U,S,V = pca(X)
z = project_data(X,U,100)
x_recovered = recover_data(z,U,100)
face = np.reshape(x_recovered[50,:],(32,32))
plt.imshow(face)
plt.show()