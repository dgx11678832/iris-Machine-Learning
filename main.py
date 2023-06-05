import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris
from sklearn.preprocessing import label_binarize

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN


def get_data():
    iris = load_iris()
    data = iris.data
    result = iris.target
    for i in range(len(data)):
        if result[i] == 0:
            class_ = "Setosa"
        elif result[i] == 1:
            class_ = "Versicolour"
        else:
            class_ = "Virginica"
        # print('花萼长度:{} 花萼宽度:{} 花瓣长度:{} 花瓣宽度:{}   类别 {}'.format(data[i][0],data[i][1],data[i][2],data[i][3], class_))

    return data, result


def visual(data, result):
    # 点图
    fig, ax = plt.subplots()
    data_sepal = data[:, :2]
    scatter = ax.scatter(data_sepal[:, 0], data_sepal[:, 1], c=result, cmap=plt.cm.gnuplot)
    a, b = scatter.legend_elements()
    b = ['$\\mathdefault {Setosa}$',
         '$\\mathdefault{Versicolour}$',
         '$\\mathdefault{Virginica}$']
    ax.legend(a, b, title="Classes")
    # plt.xlabel('萼片 长度')
    # plt.ylabel('萼片 宽度')
    plt.xlabel('Sepals Length')
    plt.ylabel('Sepals Width')
    plt.savefig('Sepals.jpg')
    plt.show()


    fig, ax = plt.subplots()
    data_sepal = data[:, 2:4]
    scatter = plt.scatter(data_sepal[:, 0], data_sepal[:, 1], c=result, cmap=plt.cm.gnuplot)
    a, b = scatter.legend_elements()
    b = ['$\\mathdefault {Setosa}$',
         '$\\mathdefault{Versicolour}$',
         '$\\mathdefault{Virginica}$']
    # plt.xlabel('花瓣 长度')
    # plt.ylabel('花瓣 宽度')
    ax.legend(a, b, title="Classes")
    plt.xlabel('Petal Length')
    plt.ylabel('Petal width')
    plt.savefig('Petal.jpg')
    plt.show()

    #


def visual2(data):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
    label_dict = ["Setosa", "Versicolour", "Virginica"]
    feature_dict = ['Sepals Length', 'Sepals Width', 'Petal Length', 'Petal width']
    data2 = []
    data2.append(data[0:50, :])
    data2.append(data[50:100, :])
    data2.append(data[100:150, :])

    for ax, cnt in zip(axes.ravel(), range(4)):

        # set bin sizes
        min_b = math.floor(np.min(data[:, cnt]))
        max_b = math.ceil(np.max(data[:, cnt]))
        bins = np.linspace(min_b, max_b, 25)

        # plottling the histograms
        for lab, col in zip(range(3), ('blue', 'red', 'green')):
            ax.hist(data2[lab][:, cnt],
                    color=col,
                    label='%s' % label_dict[lab],
                    bins=bins,
                    alpha=0.5, )
        ylims = ax.get_ylim()

        # plot annotation
        leg = ax.legend(loc='upper right', fancybox=True, fontsize=8)
        ax.set_ylim([0, max(ylims) + 2])
        ax.set_xlabel(feature_dict[cnt])
        ax.set_title('Iris histogram #%s' % str(cnt + 1))

        # adding horizontal grid lines
        ax.yaxis.grid(True)

        # hide axis ticks
        ax.tick_params(axis="both", which="both", bottom="off", top="off",
                       labelbottom="on", left="off", right="off", labelleft="on")

        # remove axis spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
    axes[0][0].set_ylabel('count')
    axes[1][0].set_ylabel('count')

    fig.tight_layout()
    plt.savefig('hist.jpg')
    plt.show()



def draw_roc(y_predict, Y_test, name=None):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):
        fpr[i], tpr[i], _ = metrics.roc_curve(Y_test[:, i], y_predict[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(Y_test.ravel(), y_predict.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.legend(loc="lower right")
    plt.title("Precision-Recall")
    plt.savefig('./{}.jpg'.format(name))
    plt.show()


def Evaluation_Indicators(y_predict, Y_test, name):
    acc = metrics.accuracy_score(Y_test, y_predict)
    print("acc得分：", acc)
    f1 = metrics.f1_score(Y_test, y_predict, average='macro')
    print("F1得分：", f1)
    print(classification_report(Y_test, y_predict))
    draw_roc(y_predict, Y_test, name)


def plt_Cluster(data, y_train, num, name):
    y_train.rename('res', inplace=True)
    print(y_train)
    result = pd.concat([pd.DataFrame(data), y_train], axis=1)
    print(result)
    Category_one = result[result['res'].values == 0]
    k1 = result.iloc[Category_one.index]
    # print(k1)
    Category_two = result[result['res'].values == 1]
    k2 = result.iloc[Category_two.index]
    # print(k2)
    Category_three = result[result['res'].values == 2]
    k3 = result.iloc[Category_three.index]
    plt.scatter(data[:50, 2], data[:50, 3], label='setosa', marker='o', c='yellow')
    plt.scatter(data[50:100, 2], data[50:100, 3], label='versicolor', marker='o', c='green')
    plt.scatter(data[100:, 2], data[100:, 3], label='virginica', marker='o', c='blue')
    # 机器学习后数据的特征散点图
    if num == 3:
        plt.scatter(k1.iloc[:, 2], k1.iloc[:, 3], label='cluster_one', marker='+', c='brown')
        plt.scatter(k2.iloc[:, 2], k2.iloc[:, 3], label='cluster_two', marker='*', c='red')
        plt.scatter(k3.iloc[:, 2], k3.iloc[:, 3], label='cluster_three', marker='x', c='black')
    elif num == 2:
        plt.scatter(k1.iloc[:, 2], k1.iloc[:, 3], label='cluster_one', marker='+', c='brown')
        plt.scatter(k2.iloc[:, 2], k2.iloc[:, 3], label='cluster_two', marker='*', c='red')
    plt.xlabel('petal length')  # 花瓣长
    plt.ylabel('petal width')  # 花瓣宽
    plt.title("result of {}".format(name))
    plt.legend()
    plt.savefig('./{}_{}.jpg'.format(name, num))
    plt.show()


def Decision_Tree_Classifier(X_train, X_test, Y_train, Y_test):
    # criterion='gini' 特征选择标准:基尼系数
    # splitter：best 特征划分标准:在特征的所有划分点中找出最优的划分点
    # max_depth: None 决策树最大深度:默认值None
    # min_samples_split:2 内部节点（即判断条件）再划分所需最小样本数 :2
    # min_samples_leaf:1  叶子节点（即分类）最少样本数 :1
    # min_weight_fraction_leaf: 0 叶子节点（即分类）最小的样本权重和:0
    # max_features:None, 在划分数据集时考虑的最多的特征值数量
    # random_state:None, 最大叶子节点数
    # max_leaf_nodes:None, 节点划分最小不纯度
    # min_impurity_decrease:0.0, 信息增益的阀值
    # class_weight:None, 类别权重
    # ccp_alpha:0.0,

    model = DecisionTreeClassifier(criterion='gini', min_samples_split=2)
    model.fit(X_train, Y_train)
    # print(tree.plot_tree(model))
    score = model.score(X_test, Y_test)  # 模型得分
    print("模型得分：", score)
    y_predict = model.predict(X_test)

    y_predict = model.predict(X_test)
    y_predict = label_binarize(y_predict, classes=[0, 1, 2])
    Y_test = label_binarize(Y_test, classes=[0, 1, 2])
    Evaluation_Indicators(y_predict, Y_test, 'Decision_Tree')


def my_Adaboost(X_train, X_test, Y_train, Y_test):
    # estimator :DecisionTreeClassifier 弱分类学习器或者弱回归学习器
    # learning_rate: 1
    # algorithm：SAMME和SAMME.R 两者的主要区别是弱学习器权重的度量，SAMME对样本集分类效果作为弱学习器权重，SAMME.R使用对样本集分类的预测概率大小来作为弱学习器权重
    model = AdaBoostClassifier(DecisionTreeClassifier(criterion='gini', min_samples_split=2), algorithm="SAMME.R")
    model.fit(X_train, Y_train)
    score = model.score(X_test, Y_test)  # 模型得分
    print("模型得分：", score)
    y_predict = model.predict(X_test)
    y_predict = label_binarize(y_predict, classes=[0, 1, 2])
    Y_test = label_binarize(Y_test, classes=[0, 1, 2])
    Evaluation_Indicators(y_predict, Y_test, 'Adaboost')


def my_Svm(X_train, X_test, Y_train, Y_test):
    # C ：惩罚参数，通常默认为1。 C越大，表明越不允许分类出错，但是C越大可能会造成过拟合，泛化效果太低。
    #                          C越小，正则化越强，分类将不会关注分类是否正确的问题，只要求间隔越大越好，此时分类也没有意义。
    # kernel（核函数）：rbf 高斯径向核  函数的引入是为了解决线性不可分的问题，将分类点映射到高维空间中以后，转化为可线性分割的问题
    # degree：当使用多项式函数作为svm内部的函数的时候，给定多项式的项数，默认为3;
    # gamma：当SVM内部使用poly、rbf、sigmoid的时候，核函数的系数值，当默认值为auto的时候，实际系数为1/n_features;
    # coef0：当核函数为poly或者sigmoid的时候，给定的独立系数，默认为0;
    # probability：是否启用概率估计，默认不启动，不太建议启动;
    # shrinking：是否开启收缩启发式计算，默认为True;
    # tol: 模型构建收敛参数，当模型的的误差变化率小于该值的时候，结束模型构建过程，默认值:1e-3;
    # cache_size：在模型构建过程中，缓存数据的最大内存大小，默认为空，单位MB;
    # class_weight：给定各个类别的权重，默认为空;
    # max_iter：最大迭代次数，默认-1表示不限制;
    # decision_function_shape：决策函数，可选值：ovo和ovr，默认为None；推荐使用ovr；1.7以上版本才有。

    model = SVC()
    model = LinearSVC()
    model.fit(X_train, Y_train)
    # print(tree.plot_tree(model))
    score = model.score(X_test, Y_test)  # 模型得分
    print("模型得分：", score)
    y_predict = model.predict(X_test)

    y_predict = model.predict(X_test)
    y_predict = label_binarize(y_predict, classes=[0, 1, 2])
    Y_test = label_binarize(Y_test, classes=[0, 1, 2])
    Evaluation_Indicators(y_predict, Y_test, 'Svm')


def Naive_Bayes(X_train, X_test, Y_train, Y_test):
    # priors=None, 类别的先验概率。如果指定了，那么先验参数就不会根据数据进行调整。
    # var_smoothing=1e-9 所有特征中最大方差的部分，为了计算的稳定性，将其加入到最大方差部分，以保证计算的稳定性。
    model = GaussianNB()  # 高斯贝叶斯分类器
    model.fit(X_train, Y_train)
    # print(tree.plot_tree(model))
    score = model.score(X_test, Y_test)  # 模型得分
    print("模型得分：", score)
    y_predict = model.predict(X_test)

    y_predict = model.predict(X_test)
    y_predict = label_binarize(y_predict, classes=[0, 1, 2])
    Y_test = label_binarize(Y_test, classes=[0, 1, 2])
    Evaluation_Indicators(y_predict, Y_test, 'Naive_Bayes')


def my_KMeans(data, num):
    # n_clusters=3,
    # init="k-means++",
    # n_init="warn",
    # max_iter=300,
    # tol=1e-4,
    # verbose=0,
    # random_state=None,
    # copy_x=True,
    # algorithm="lloyd",
    model = KMeans(n_clusters=num)
    model.fit(data)

    y_train = pd.Series(model.labels_)
    plt_Cluster(data, y_train, num, 'KMeans')


def Hierarchical_Clustering(data, num):
    # n_clusters = 3,
    # *,
    # affinity = "deprecated",  # TODO(1.4): Remove
    # metric = None,  # TODO(1.4): Set to "euclidean"
    # memory = None,
    # connectivity = None,
    # compute_full_tree = "auto",
    # linkage = "ward",
    # distance_threshold = None,
    # compute_distances = False
    model = AgglomerativeClustering(n_clusters=num)
    model.fit(data)
    y_train = pd.Series(model.labels_)
    plt_Cluster(data, y_train, num, 'Hierarchical_Clustering')


def my_GaussianMixture(data, num):
    # n_components = 3,
    # *,
    # covariance_type = "full",
    # tol = 1e-3,
    # reg_covar = 1e-6,
    # max_iter = 100,
    # n_init = 1,
    # init_params = "kmeans",
    # weights_init = None,
    # means_init = None,
    # precisions_init = None,
    # random_state = None,
    # warm_start = False,
    # verbose = 0,
    # verbose_interval = 10,
    model = GaussianMixture(n_components=num)
    model.fit(data)
    labels = model.predict(data)
    y_train = pd.Series(data=labels)
    plt_Cluster(data, y_train, num, 'GaussianMixture')


def my_DBscan(data, num):
    # eps = 0.5,
    # *,
    # min_samples = 3,
    # metric = "euclidean",
    # metric_params = None,
    # algorithm = "auto",
    # leaf_size = 30,
    # p = None,
    # n_jobs = None,
    model = DBSCAN(min_samples=num)  # 构造聚类器,一个参数是半径，一个是密度
    model.fit(data)
    y_train = pd.Series(model.labels_)  # 有-1
    # print(y_train.shape)
    # print(y_train)
    plt_Cluster(data, y_train, num, "DBscan")


def classify_iris(X_train, X_test, Y_train, Y_test):
    ## 决策树
    print('Decision_Tree_Classifier')
    Decision_Tree_Classifier(X_train, X_test, Y_train, Y_test)  # criterion='gini', min_samples_split=2
    ## 自适应增强算法Adaboost
    print('Adaboost')
    my_Adaboost(X_train, X_test, Y_train, Y_test)
    ## SVM
    print('SVM')
    my_Svm(X_train, X_test, Y_train, Y_test)
    ## Naïve_Bayes 朴素贝叶斯
    print('朴素贝叶斯')
    Naive_Bayes(X_train, X_test, Y_train, Y_test)


def Cluster_iris(data, num=3):
    # KMeans
    my_KMeans(data, num)
    # Hierarchical Clustering
    Hierarchical_Clustering(data, num)
    # DBscan
    my_DBscan(data, num)
    # Gaussian Mixture Model
    my_GaussianMixture(data, num)


if __name__ == '__main__':
    data, result = get_data()
    # 点图
    # visual(data, result)
    # 直方图
    # visual2(data)
    # 划分数据集
    X_train, X_test, Y_train, Y_test = train_test_split(data, result, test_size=0.2, random_state=75)
    # #分类
    # classify_iris(X_train, X_test, Y_train, Y_test)
    # #聚类
    Cluster_iris(data,3)
