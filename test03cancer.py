# 导入威斯康星乳腺肿瘤数据集
import numpy as np
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=38)

# 使用高斯贝叶斯进行建模
from sklearn.naive_bayes import GaussianNB

gs = GaussianNB().fit(X_train, y_train)
# print("模型得分:{:.2f}".format(gs.score(X_test, y_test)))
#
# # 预测使用第312个样本值预测
# print("模型的预测分类为:{}".format(gs.predict([X[312]])))
# print("样本的分类为{}".format(y[312]))

# 学习曲线
from sklearn.model_selection import learning_curve  # 导入学习曲线库
# 导入拆分工具
from sklearn.model_selection import ShuffleSplit
# 定义一个函数绘制学习曲线
import matplotlib.pyplot as plt

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)

    # 设定横轴标签
    plt.xlabel("Training examples")
    # 设定纵轴标签
    plt.ylabel("Score")

    train_sizes, tr_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(test_scores, axis=1)
    plt.grid()

    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, train_scores_mean, '*-', color='g', label='Cross-vaildation score')

    plt.legend(loc='lower right')
    return plt

if __name__=='__main__':

# 设定图题
    title = "Learning Curves (Naive Bayes)"
    # 设定拆分数量
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    # 设定模型为高斯朴素贝叶斯
    estimator = GaussianNB()

    # 调用函数
    plot_learning_curve(estimator, title, X, y, ylim=(0.9, 1.01), cv=cv, n_jobs=4)

    # 显示图片
    plt.show()
