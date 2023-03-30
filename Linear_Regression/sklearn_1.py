'''
Scikit-learn(sklearn)是机器学习中常用的第三方模块，
对常用的机器学习方法进行了封装，包括回归(Regression)、
降维(Dimensionality Reduction)、
分类(Classfication)、聚类(Clustering)等方法
'''

from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
#画三维
from mpl_toolkits.mplot3d import Axes3D
import os

#保存图片
PROJECT_ROOT_DIR="."
MODEL_ID="linear_models"
def save_pic(pic_id):
    path = os.path.join(PROJECT_ROOT_DIR,"image",MODEL_ID,pic_id+".png")
    print("SAVE PICTURE",pic_id)
    plt.savefig(path,format="png")

x = 2*np.random.rand(100,1)#100行数据
y = 4+3*x+np.random.rand(100,1)

model = LinearRegression()
model.fit(x,y)
plt.scatter(x,y,c="b",s=20)
np.random.seed(0)
plt.plot(x,model.predict(x),color="red")
#save_pic("skelearn_res")
plt.show()