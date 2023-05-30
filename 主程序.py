import numpy as np 
import pandas as pd 
import os
from collections import Counter
from sklearn.metrics import log_loss
df_train = pd.read_csv('C:/Users/DELL/Desktop/毕设/数据集/train.csv')
#df_train.head()

df_test = pd.read_csv('C:/Users/DELL/Desktop/毕设/数据集/test.csv')
#df_test.head()

print(df_train['is_duplicate'].mean() )

print('均值:', log_loss(df_train['is_duplicate'], np.zeros_like(df_train['is_duplicate']) + df_train['is_duplicate'].mean()))

#valid=pd.read_csv('C:/Users/DELL/Desktop/毕设/数据集/valid.csv')

train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
test_qs = pd.Series(df_test['question1'].tolist() + df_test['question2'].tolist()).astype(str)
#合并问题并转换数据类型

dist_train = train_qs.apply(lambda x: len(x.split(' ')))
dist_test = test_qs.apply(lambda x: len(x.split(' ')))
#把两个问题的单词用空格拆分

import nltk
from nltk.corpus import stopwords
stops = set(stopwords.words("english"))

def word_match_share(row):
#查找共同单词
    q1words = {}
    q2words = {}
    #print(row)
    
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1 #键赋值：按列标记不在stopeword的词
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        return 0
    
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    
    #查看字典中的每个单词，并检查是否在两个字典中都找到了它，列表存储
    #shared_words_in_q1和shared_words_in_q2内容一样
    
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    #dice
    #交集两倍除以总数
    #R=问题一和问题二的公共单词数量/两列问题中有意义词的总量
    #计算一个单词在问题中单独出现的次数，并将该数字除以两个词典中的单词总数
    #得出给定单词在两个问题中单独出现频率的平均百分比。    
    return R

train_word_match = df_train.apply(word_match_share, axis=1, raw=False)
#raw : 布尔型, 默认为：False
#如果为False : 将每个行或列作为一个Series传递给函数。

print('00000000')
print(train_word_match)

def get_weight(count, eps=10000, min_count=2):
    #在文档中出现概率低，说明这个词具有良好的类别区分能力，应赋予其更高的权重。
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)

#eps = 5000 
words = (" ".join(train_qs)).lower().split()
print(words)
#保持空格连接成长字符串
counts = Counter(words)
#统计列表同一行两个问题的单词，将元素数量统计，然后计数并返回一个字典，键为元素，值为元素个数。

weights = {word: get_weight(count) for word, count in counts.items()}
#每个单词初始化一个键值对，键是单词的字符串表示，值为get_weight计算权重


import nltk
from nltk.corpus import stopwords


stops = set(stopwords.words("english"))
def tfidf_word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    if total_weights == 0:
        return 0
    
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    #weights是字典，键是单词的字符串表示，值为get_weight计算权重，get方法返回值
    #total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    
    #get(w, 0)返回键w的值指定键的值不存在时，返回该默认值0
    
    R = np.sum(shared_weights) / np.sum(total_weights)

    return R

tfidf_train_word_match = df_train.apply(tfidf_word_match_share, axis=1, raw=False)
print ('111111111')
print(tfidf_train_word_match)

x_train = pd.DataFrame()
x_test = pd.DataFrame()
x_train['word_match'] = train_word_match
x_train['tfidf_word_match'] = tfidf_train_word_match
x_test['word_match'] = df_test.apply(word_match_share, axis=1, raw=False)
x_test['tfidf_word_match'] = df_test.apply(tfidf_word_match_share, axis=1, raw=False)
print('1')

print(x_train)
#404289
print(x_test)
#2345795
y_train = df_train['is_duplicate'].values
pos_train = x_train[y_train == 1]
#pos_train包含x_train中的所有单词，其中y为1
neg_train = x_train[y_train == 0]
#neg-train包含所有相同的单词，但y为0
print('2')

print(pos_train)
#149263
print('前')
print(neg_train)
#255027


neg_train = pd.concat([neg_train, neg_train, neg_train])
    #自拼接来放大负例子的个数（负采样）
    #pd.concat拼接表格



print('后')
print(neg_train)
#631223
print(len(pos_train) / (len(pos_train) + len(neg_train)))

x_train = pd.concat([pos_train, neg_train])
y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
#y_train是两个数据集中重复的单词列表
#np.zeros返回来一个给定形状和类型的用0填充的数组
print('3')
print(len(y_train ))
#print(x_train)
#print(y_train)
from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)


import xgboost as xgb

params = {}
params['objective'] = 'binary:logistic'
#二分类逻辑回归模型
params['eval_metric'] = 'logloss'
#验证数据的评估指标
params['eta'] = 0.02
#学习率
params['max_depth'] = 4

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)
print('4')
print(d_train)
print (d_valid)
#将训练集和测试集构造为xgboost中可以使用的格式

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

bst = xgb.train(params, d_train, num_boost_round=10000000, evals=watchlist, early_stopping_rounds=50, verbose_eval=10)
#11549
d_test = xgb.DMatrix(x_test)
p_test = bst.predict(d_test)

print(len(p_test))
print(len(df_test['is_duplicate']))









