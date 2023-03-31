#k-means
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio

# 寻找最近的聚类中心
def find_cloest_centroids(X, centroids):
    m = X.shape[0]
    k = centroids.shape[0]
    index = np.zeros(m)

    for i in range(m):
        min_dist = 1000000
        for j in range(k):
            dist = np.sum(np.power(X[i,:]-centroids[j,:],2))
            #dist = np.sum((X[i, :] - centroids[j, :]) ** 2)
            if dist < min_dist:
                min_dist = dist
                index[i] = j  # 保存聚类中心
    return index

#计算聚类中心
def compute_centroids(X,index,k):
    m,n = X.shape
    centroids = np.zeros((k,n))
    for i in range(k):
        indices = np.where(index==i)
        len = np.size(indices)
        centroids[i] = (np.sum(X[indices,:],axis=1)/len).ravel()
    return centroids

#迭代
def run_k_means(X,init_centroids,max_iter):
    m,n = X.shape
    k = init_centroids.shape[0]
    index = np.zeros(m)
    centroids = init_centroids
    for i in range(max_iter):
        index = find_cloest_centroids(X,centroids)
        centroids = compute_centroids(X,index,k)
    return index,centroids

#随机初始化聚类中心
def init_centroids(X,k):
    m,n = X.shape
    centroids = np.zeros((k,n))
    index = np.random.randint(0,m,k)
    for i in range(k):
        centroids[i] = X[index[i],:]
    return centroids

if __name__ == '__main__':

    #TODO 准备数据
    data = sio.loadmat('ex7data2.mat')
    X = data['X']
    data2 = pd.DataFrame(data.get('X'),columns=['X1','X2'])

    #TODO 初始化聚类中心
    initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
    index,centroids = run_k_means(X,initial_centroids,50)

    c1 = X[np.where(index==0)[0],:]
    c2 = X[np.where(index==1)[0],:]
    c3 = X[np.where(index==2)[0],:]

    fig,ax = plt.subplots(figsize = (12,8))
    ax.scatter(c1[:,0],c1[:,1],s=20,c='r',label="C1")
    ax.scatter(c2[:,0],c2[:,1],s=20,c='g',label="C2")
    ax.scatter(c3[:,0],c3[:,1],s=20,c='b',label="C3")
    ax.legend()
    #plt.savefig('exdata2.png')
    plt.show()




