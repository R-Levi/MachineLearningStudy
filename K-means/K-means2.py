#将K-means应用于图像压缩
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
'''
pic = plt.imread('curry.jpg')
plt.imshow(pic)
plt.show()
'''


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

'''
img = plt.imread('curry.jpg')
sio.savemat('out_curry.mat',{'A':img})
'''



image_data = sio.loadmat('out_curry.mat')
A = image_data['A']
#print(A.shape)#(128, 128, 3)

#归一化
A = A/255
#变成128*12*行3列
X = np.reshape(A,(A.shape[0]*A.shape[1],A.shape[2]))
#随机初始化聚类中心
centroids = init_centroids(X, 16)
#迭代
index,centroids = run_k_means(X,centroids,10)
index = index.astype(int)
#每个像素与聚类中心匹配
x_recovered = np.zeros((len(index),3))
for i in range(len(index)):
    #print(index[i])
    x_recovered[i] = centroids[index[i],:]

# reshape to the original dimensions
x_recovered = np.reshape(x_recovered, (A.shape[0], A.shape[1], A.shape[2]))
plt.imshow(x_recovered)
#plt.savefig('bird_small_res.png')
plt.show()

