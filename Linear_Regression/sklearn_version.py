from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
import numpy as np
import matplotlib.pyplot as plt
#画三维
from mpl_toolkits.mplot3d import Axes3D
import os
np.random.seed(1024)

#保存图片
PROJECT_ROOT_DIR="."
MODEL_ID="linear_models"
def save_pic(pic_id):
    path = os.path.join(PROJECT_ROOT_DIR,"image",MODEL_ID,pic_id+".png")
    print("SAVE PICTURE",pic_id)
    plt.savefig(path,format="png")

#TODO 准备数据
# #多元回归
# boston = load_boston()
# x = boston['data']
# y = boston['target']
# 一元
x = np.random.randn(100,1)
y = 3*x+np.random.randn(100,1)
#TODO 准备模型
model = LinearRegression()
model.fit(x,y)
plt.scatter(x,y,c="b",s=20)
plt.plot(x,model.predict(x),color="red")
#save_pic("skelearn_res")
plt.show()