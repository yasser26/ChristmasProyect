# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.cluster import KMeans

def main():

    iris = datasets.load_iris()
    # iris

    iris_data = pd.DataFrame(iris.data, columns=iris['feature_names'])
    iris_target = pd.DataFrame(iris.target, columns=['target'])

    #print iris_target
    #print iris_data

    def map_target(target_num):
        return iris.target_names[int(target_num)]

    iris_target_name = iris_target.apply(map_target,1)

    #print iris_target_name

    kmeans_3 = KMeans(n_clusters=3)
    kmeans_iris = kmeans_3.fit(iris_data[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)']])

    fig = plt.figure(1, figsize=(12,9))
    ax = Axes3D(fig, rect=[0,0,.95,1], elev=48, azim=134)
    labels = kmeans_iris.labels_
    ax.scatter(iris_data.ix[:,3], iris_data.ix[:,0], iris_data.ix[:,2], c=labels.astype(np.float))

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')

    plt.show()

    print(pd.Series([(iris_target.loc[i][0], kmeans_iris.labels_[i]) for i in range(len(iris.target))]).value_counts())
    print(pd.Series([(iris_target_name.loc[i][0], kmeans_iris.labels_[i]) for i in range(len(iris.target))]).value_counts())

def elbow_plot(data, maxK=10, seed_centroids=None):
    """
        parameters:
        - data: pandas DataFrame (data to be fitted)
        - maxK (default = 10): integer (maximum number of clusters with which to run k-means)
        - seed_centroids (default = None ): float (initial value of centroids for k-means)
    """
    sse = {}
    for k in range(1, maxK):
        print("k: ", k)
        if seed_centroids is not None:
            seeds = seed_centroids.head(k)
            kmeans = KMeans(n_clusters=k, max_iter=500, n_init=100, random_state=0, init=np.reshape(seeds, (k,1))).fit(data)
            data["clusters"] = kmeans.labels_
        else:
            kmeans = KMeans(n_clusters=k, max_iter=300, n_init=100, random_state=0).fit(data)
            data["clusters"] = kmeans.labels_
        # Inertia: Sum of distances of samples to their closest cluster center
        sse[k] = kmeans.inertia_
    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.show()
    return


iris = datasets.load_iris()
iris_data = pd.DataFrame(iris.data, columns=iris['feature_names'])

elbow_plot(iris_data[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)']], maxK=10)



if __name__ == '__main__':
    main()
