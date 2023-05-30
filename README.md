# Undergraduate-graduation-project（高建煜毕业设计代码说明）
##摘要（abstract）

摘要
随着互联网的普及和在线问答社区的兴起，相似提问识别成为了自然语言处理领域的一个重要任务。本文旨在研究基于机器学习的相似提问识别方法，以减少重复问题的数量，提高问答系统的效率。本研究分为三个部分：数据集分析和处理、特征表示方法和分类器选择。首先，对数据集进行分析，考察各个部分的数据体量和分布特征并调整数据不同类别样本的比例，在训练前从训练集拆分出验证集。其次，基于TF-IDF算法思想，结合Dice系数计算两个提问之间的相似度。最后，基于该相似度训练XGBoost模型，预测输入的两个提问是否相似，以完成相似提问识别任务。实验针对现实情况调整数据，进行问题相似度的计算，训练模型，并使用Kaggle后台数据测试模型。本研究为相似提问识别构建了有效的机器学习方法，有助于进一步提高在线问答系统的效率，为用户提供更加准确的解答。

关键词：机器学习；自然语言处理；逆文档频率；梯度提升算法


With the popularity of the Internet and the rise of online Q&A communities, similar question recognition has become an important task in the field of natural language processing. This thesis aims to study machine learning-based similar question recognition methods to reduce the number of repetitive questions and improve the efficiency of Q&A systems. This study is divided into three parts: dataset analysis and processing, feature representation methods, and classifier selection. First, the dataset is analyzed to examine the data volume and distribution characteristics of each part and adjust the proportion of different categories of data samples. Before training, a validation set is split from the training set. Secondly, based on the TF-IDF algorithm idea and combined with the Dice coefficient, the similarity between two questions is calculated. Finally, based on this similarity, an XGBoost model is trained to predict whether two input questions are similar to complete the task of similar question recognition. The experiment adjusts data according to real situations, calculates problem similarity, trains models, and tests models using Kaggle backend data. This study constructs an effective machine learning method for similar question recognition, which helps to further improve the efficiency of online Q&A systems and provide users with more accurate answers.

Keyword: machine learning; natural language processing; inverse document frequency; gradient boosting algorithm

##(Code Description)代码说明

代码按照模块功能描述有以下顺序：
导入相关库；
打开数据文件并转换格式；
打开停用词方法；
定义函数逐行寻找共有词，并根据共有词数量返回问题对Dice相似度；
定义并使用词权重计算函数；
定义函数逐行寻找带权共有词，并根据共有词权重返回问题对Dice相似度；
构造数据结构收集之前函数的计算结果；
正例和负例分开；
复制负例；
将训练数据转换为可以输入分类器的格式；
拆分训练集和测试集；
设置分类器参数，训练模型并预测测试集数据；
将预测结果生成为Kaggle项目所要求的格式的文件。

The code is described in the following order according to module functions:
Import related libraries;
Open data file and convert format;
Open Stop Word Method;
Define a function to find common words line by line and return the Dice similarity of the problem based on the number of common words;
Define and use word weight calculation functions;
Define a function to search for weighted common words row by row, and return the Dice similarity of the problem based on the weight of the common words;
Construct a data structure to collect the calculation results of the previous function;
Separate positive and negative examples;
Copy negative examples;
Convert training data into a format that can be input into a classifier;
Split training and testing sets;
Set classifier parameters, train models, and predict test set data;
Generate the predicted results into a file in the required format for the Kaggle project.

##代码运行说明（Code operation instructions）

该项目中的代码“主程序.py”是本地运行，如需拷贝使用，除了对应安装库以外还需要在Kaggle项目Quora Question Pairs（https://www.kaggle.com/competitions/quora-question-pairs/overview）中下载训练集数据，并对应修改代码中的读取路径。
当然也可以以移步到我在Kaggle网站的笔记（https://www.kaggle.com/code/gaojianyu/quora-question-pairs/notebook），那里可以直接运行。

The code "main program. py" in this project runs locally. If you need to copy and use it, in addition to the corresponding installation library, you also need to run it in the Kaggle project Quora Question Pairs（ https://www.kaggle.com/competitions/quora-question-pairs/overview ）Download the training set data and modify the read path in the code accordingly.

Of course, you can also move to my notes on the Kaggle website（[https://www.kaggle.com/code/gaojianyu/quora-question-pairs](https://www.kaggle.com/code/gaojianyu/quora-question-pairs)），where the code can be run directly.



