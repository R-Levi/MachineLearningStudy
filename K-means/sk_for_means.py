import numpy as np
from sklearn.cluster import KMeans,DBSCAN,MeanShift
from sklearn import metrics
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
# 定义数据集
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)

model1 = KMeans(n_clusters=3)
model1.fit(X)
pre_y1 = model1.predict(X)
cls1 = np.unique(pre_y1)
for c in cls1:
    idx = np.where(pre_y1==c)
    plt.scatter(X[idx,0],X[idx,1],cmap=c)
plt.show()
completeness_score = metrics.cluster.completeness_score(y,pre_y1)
print(completeness_score)

model2 = MeanShift()
model2.fit(X)
pre_y2 = model1.predict(X)
cls2 = np.unique(pre_y2)
for c in cls2:
    idx = np.where(pre_y2==c)
    plt.scatter(X[idx,0],X[idx,1],cmap=c)
plt.show()
completeness_score = metrics.cluster.completeness_score(y,pre_y2)
print(completeness_score)

model3 = DBSCAN(eps=0.5,min_samples=50)
pre_y3  = model3.fit_predict(X)
cls3 = np.unique(pre_y3)
for c in cls3:
    idx = np.where(pre_y3==c)
    plt.scatter(X[idx,0],X[idx,1],cmap=c)
plt.show()
completeness_score = metrics.cluster.completeness_score(y,pre_y3)
print(completeness_score)
