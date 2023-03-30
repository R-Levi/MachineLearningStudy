#softmax 实现多分类 对数似然函数
import torch
import scipy.io as sio
import numpy as np

def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition

def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])

def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def train_epoch(net, train_iter, loss, updater):  #@save
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    all_loss = []
    train_acc = []
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        all_loss.append(l)
        train_acc.append(accuracy(y_hat, y))
    return

if __name__ == '__main__':
    # 读取matlab的数据集，包括是20*20的像素以及0-9标签
    matinfo = sio.loadmat("ex3data1.mat")
    X = matinfo['X']  # 20*20像素
    Y = matinfo['y'][:, 0]  # 标签

    num_inputs = 400
    num_outputs = 10
    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)

    model = net()
    loss = cross_entropy()
    updater = torch.optim.SGD((W,b),lr=0.001)
    for epoch in range(10):
        train_metrics = train_epoch(model, X,Y, loss, updater)