import numpy as np
#导入贝叶斯
from sklearn.naive_bayes import BernoulliNB

X=np.array([[0,1,0,1],
           [1,1,1,0],
           [0,1,1,0],
           [0,0,0,1],
           [0,1,1,0],
           [0,1,0,1],
           [1,0,0,1]])
y=np.array([0,1,1,0,1,0,0])
#对不同的分类计算每个特征为1的数量
# counts={}
# for label in np.unique(y):
#     counts[label]=X[y==label].sum(axis=0)
#
#
# #打印结果
# print("feature counts:\n{}".format(counts))
clf=BernoulliNB().fit(X,y)
#要进行预测的这一天，没有刮北风，也不闷热
#但是多云，天气预报没有说下雨
Next_Day=[[0,0,1,0]]
pre=clf.predict(Next_Day)

if pre==[1]:
    print("要下雨了，快收衣服")
else:
    print("放心，又是一个艳阳天")
print(clf.predict_proba(Next_Day))