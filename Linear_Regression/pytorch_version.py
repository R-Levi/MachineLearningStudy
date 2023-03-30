import torch
import torch.nn as nn
from sklearn.datasets import load_boston

class LinearModel(nn.Module):
    def __init__(self,ndim):
        super(LinearModel, self).__init__()
        self.ndim = ndim
        self.weight = nn.Parameter(torch.randn(ndim,1)) #权重
        self.bias = nn.Parameter(torch.randn(1)) #偏置

    def forward(self,x):
        return x.mm(self.weight)+self.bias,self.weight,self.bias

device = "cuda" if torch.cuda.is_available() else "cpu"
#TODO 准备数据、模型、优化器、损失函数
boston = load_boston() #(506,13),(506,)
model = LinearModel(13)
loss = nn.MSELoss()
optim = torch.optim.SGD(model.parameters(),lr=1e-6)
X = torch.tensor(boston['data'],dtype=torch.float32)
Y = torch.tensor(boston['target'],dtype=torch.float32)
model.to(device)
X,Y = X.to(device),Y.to(device)
#TODO 开始训练
for step in range(100000):
    optim.zero_grad()
    predict, weight, bias = model(X)
    l = loss(predict, Y)
    if step%1000 == 0:
        print("MSE Loss：{:.3f}".format(l.item()))
    l.backward()
    optim.step()


