import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

class naiveBayesClassfier():
    '''
    fit: 训练分类器
    predict：预测分类结果
    '''
    def __init__(self):
        # 先验概率
        self.prio_prob = {}
        # 似然函数
        self.liklihood_prob = {}
        # 证据因子
        self.evidence_prob = {}

    def fit(self, data):
        '''
        - data: 输入数据
        - label: 预测值
        return: 概率 '''
        self.feature = data.columns.values
        row, _ = data.shape
        label = data['label']
        # 先验概率
        cls = np.unique(label)
        self.cls = cls
        for c in cls:
            self.prio_prob[c] = len(np.where(label==c)[0])/row

        # 似然
        for l in cls:
            data_n = data[data['label']==l]
            row_n, _ = data_n.shape
            prob2 = {}
            for col in self.feature:
                prob1 = {}
                for i, j in data_n[col].value_counts().to_dict().items():
                    prob1[i] = j / row_n
                prob2[col] = prob1
            self.liklihood_prob[l] = prob2

        # 证据因子
        for col in self.feature:
            prob3 = {}
            for i, j in data[col].value_counts().to_dict().items():
                prob3[i] = j / row
            self.evidence_prob[col] = prob3

    def cal_prob(self, input, res):
        '''计算似然概率，证据因子'''
        liklihood = 1
        evidence = 1
        for col in input:
            print(col)
            v = input[col]
            liklihood = liklihood * self.liklihood_prob[res][col][v]
            evidence = evidence * self.evidence_prob[col][v]

        return liklihood, evidence

    def predict(self, input):
        '''
        - input：需要预测的数据
        return：预测的结果及概率
        '''
        prob = {}
        for res in self.cls:
            liklihood, evidence = self.cal_prob(input, res)
            prob[res] = self.prio_prob[res] * liklihood / evidence

        print(prob)
        prediction = max(prob, key=lambda x: prob[x])
        probability = prob[prediction]

        return prediction, probability


if __name__ == '__main__':
    # 定义属性值
    outlook = ["晴朗", "多云", "雨天"]
    Temperature = ["高温", "中温", "低温"]
    Humidity = ["高湿", "一般"]
    Wind = ["大", "小"]
    PlayTennis = ["是", "否"]
    Play = []
    Play.append(outlook)
    Play.append(Temperature)
    Play.append(Humidity)
    Play.append(Wind)
    Play.append(PlayTennis)
    # 数据集
    data = [["晴朗", "高温", "高湿", "小", "否"],
            ["晴朗", "高温", "高湿", "大", "否"],
            ["多云", "高温", "高湿", "小", "是"],
            ["雨天", "中温", "高湿", "小", "是"],
            ["雨天", "低温", "一般", "小", "是"],
            ["雨天", "低温", "一般", "大", "否"],
            ["多云", "低温", "一般", "大", "是"],
            ["晴朗", "中温", "高湿", "小", "否"],
            ["晴朗", "低温", "一般", "小", "是"],
            ["雨天", "中温", "一般", "小", "是"],
            ["晴朗", "中温", "一般", "大", "是"],
            ["多云", "中温", "高湿", "大", "是"],
            ["晴朗", "高温", "一般", "小", "是"],
            ["多云", "高温", "一般", "小", "是"],
            ["雨天", "中温", "高湿", "大", "否"],
            ["晴朗", "中温", "高湿", "大", "否"]
            ]

    data = pd.DataFrame(data=data,columns=['outlook','Temperature','Humidity','Wind','label'])
    train,test = train_test_split(data,test_size=2)


    model = naiveBayesClassfier()
    model.fit(train)
    model.predict(test)
