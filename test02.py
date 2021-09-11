from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split


X, y = make_blobs(n_samples=500, centers=5, random_state=8)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)

from sklearn.naive_bayes import BernoulliNB
#导入高斯贝叶斯
from sklearn.naive_bayes import GaussianNB
#导入多项式朴素贝叶斯
from sklearn.naive_bayes import MultinomialNB
#导入数据预处理工具
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler().fit(X_train)
X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)


nb = BernoulliNB().fit(X_train, y_train)
gs=GaussianNB().fit(X_train,y_train)
#拟合预处理之后的数据X值非负
mnb=MultinomialNB().fit(X_train_scaled,y_train)

# print("模型得分：{:.2f}".format(mnb.score(X_test_scaled,y_test)))

import matplotlib.pyplot as plt
import numpy as np

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))
Z = mnb.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
# Z=Z.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Accent)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.cool, edgecolors='k')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.cool, marker='*', edgecolors='k')
# Z=Z.reshape(xx.shape)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Classifier:MultinomialNB")
plt.show()
