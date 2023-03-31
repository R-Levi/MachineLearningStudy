from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


wine = load_wine()

Xtrain,Xtest,Ytrain,Ytest = train_test_split(wine.data,wine.target,test_size=0.25)

tree_model = DecisionTreeClassifier(criterion='entropy')
fited = tree_model.fit(Xtrain, Ytrain)  # 训练树模型
score = fited.score(Xtest, Ytest)  # 返回预测的准确度
print("准确度：", score)
print("特征的重要性：", fited.feature_importances_)  # 查看特征的重要性

#预剪枝，设置树的高度
test = []
for i in range(10):
    clf = DecisionTreeClassifier(max_depth=i + 1, criterion="entropy", random_state=30,splitter="random")  # 实例化树模型
    clf = clf.fit(Xtrain, Ytrain)  # 训练树模型
    score = clf.score(Xtest, Ytest)  # 训练集上的准确度
    test.append(score)
print("最高精度为：", max(test), "所对应的树的深度：", test.index(max(test)) + 1)
plt.plot(range(1, 11), test, color="red", label="max_depth")
plt.legend()  # 显示标签
plt.show()