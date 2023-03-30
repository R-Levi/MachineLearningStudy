import numpy as np
'''
x = np.random.randn(100)*10
y = np.c_[np.ones((100)),x]
print(y)
'''
#在进行np.dot() 矩阵乘法时，
# 若数组左乘矩阵：则可将其作为行矢量。
# 若数组右乘矩阵，则可将数组作为列矢量。
# 相应的矩阵在右时，其最后一个维度的长度必须与数组长度一致，
# 矩阵在乘法左侧时，其第一个维度必须与矩阵的长度一致。
theta = np.zeros((10,))
x = np.ones((10,1))
y = x.dot(theta)
print(y)
print(theta.dot(x))
#print(x*theta)
#print(theta*x)


