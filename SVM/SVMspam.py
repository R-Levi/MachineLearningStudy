#使用SVM来构建垃圾邮件过滤器垃圾邮件(y=1)和非垃圾邮件(y=0)
import numpy as np
from sklearn import svm
import scipy.io as sio
import re#正则
import nltk#自然语言处理工具包


spam_train = sio.loadmat('spamTrain.mat')
spam_test = sio.loadmat('spamTest.mat')

X = spam_train['X']
Xtest = spam_test['Xtest']

y = spam_train['y'].ravel()
ytest = spam_test['ytest'].ravel()
#print(X.shape,y.shape,Xtest.shape,ytest.shape)

svc = svm.SVC(C=0.1,kernel='linear')
svc.fit(X,y)
'''
报错：DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
解决：当需要一维数组时，传递了列向量y。
把y拉成一维数组y.ravel
'''
print("训练集精确度{}%".format(np.round(svc.score(X,y)*100,2)))
print("测试集精确度{}%".format(np.round(svc.score(Xtest,ytest)*100,2)))

'''邮件预处理'''
f = open('emailSample1.txt','r').read()
#print(f)
def processEmail(email):
    email = email.lower()  # 转化为小写
    email = re.sub('<[^<>]+>', ' ', email)  # 移除所有HTML标签
    email = re.sub('(http|https)://[^\s]*', 'httpaddr', email)  # 将所有的URL替换为'httpaddr'
    email = re.sub('[^\s]+@[^\s]+', 'emailaddr', email)  # 将所有的地址替换为'emailaddr'
    email = re.sub('\d+', 'number', email)  # 将所有数字替换为'number'
    email = re.sub('[$]+', 'dollar', email)  # 将所有美元符号($)替换为'dollar'

    # 将所有单词还原为词根//移除所有非文字类型，空格调整
    stemmer = nltk.stem.PorterStemmer()  # 使用Porter算法
    tokens = re.split('[ @$/#.-:&*+=\[\]?!()\{\},\'\">_<;%]', email)  # 把邮件分割成单个的字符串,[]里面为各种分隔符
    tokenlist = []
    for token in tokens:
        token = re.sub('[^a-zA-Z0-9]', '', token)  # 去掉任何非字母数字字符
        try:  # porterStemmer有时会出现问题,因此用try
            token = stemmer.stem(token)  # 词根
        except:
            token = ''
        if len(token) < 1:
            continue  # 字符串长度小于1的不添加到tokenlist里
        tokenlist.append(token)
    return tokenlist
processed_f = processEmail(f)
#print(len(processed_f))

vocab_list = np.loadtxt('vocab.txt',dtype='str',usecols=1)
m = len(vocab_list)

'''单词序列映射'''
def word_indices(processed_f,vocab_list):
    indices = []
    for i in range(len(processed_f)):
        for j in range(len(vocab_list)):
            if(processed_f[i]!=vocab_list[j]):
                continue
            indices.append(j+1)#每个单词对英语词汇表的第几个
    return indices


f_indices = word_indices(processed_f, vocab_list)
#print(len(f_indices))
'''特征提取'''
def emailFratures(indices):
    features = np.zeros((1899))
    for i in indices:
        features[i-1] = 1
    return features
#print(sum(emailFratures(indices)))45

#预测
def pre(x,name):
    predict = svc.predict(x)
    if predict == 0:
        print("{}是非垃圾邮件".format(name))
    else:
        print("{}是垃圾邮件".format(name))

t = open('emailSample2.txt','r').read()
processed_t = processEmail(t)
t_indices = word_indices(processed_t,vocab_list)

x2 = np.reshape(emailFratures(t_indices),(1,m))
pre(x2,"email2")
x1 = np.reshape(emailFratures(f_indices),(1,m))
pre(x1,"email1")

spam1 = open('spamSample1.txt', 'r').read()
process_spam1 = processEmail(spam1)
indices_spam1 = word_indices(process_spam1, vocab_list)

spam2 = open('spamSample2.txt', 'r').read()
process_spam2 = processEmail(spam2)
indices_spam2 = word_indices(process_spam2, vocab_list)

x3 = np.reshape(emailFratures(indices_spam1),(1,m))
pre(x3,"email3")
x4 = np.reshape(emailFratures(indices_spam2),(1,m))
pre(x4,"email4")

