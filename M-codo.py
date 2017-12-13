# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.cluster import KMeans

def elbow_plot(data, n_cluster, seed_centroids=None):
	"""parameters:
	- data: pandas DataFrame (data to be fitted)
	- maxK (default = 10): integer (maximum number of clusters with which to run k-means)
	- seed_centroids (default = None ): float (initial value of centroids for k-means)"""
	"""print"k: ", k"""
	if seed_centroids is not None:
		seeds = seed_centroids.head(n_cluster)
		kmeans = KMeans(n_clusters=n_cluster, max_iter=500, n_init=100, random_state=0, init=np.reshape(seeds, (k,1))).fit(data)
	else:
		kmeans = KMeans(n_clusters=n_cluster, max_iter=300, n_init=100, random_state=0).fit(data)
		# Inertia: Sum of distances of samples to their closest cluster center
	valor = kmeans.inertia_
	return (valor)
""" plt.figure()
	plt.plot(list(sse.keys()), list(sse.values()))
	plt.show()
	return"""
iris = datasets.load_iris()
iris_data = pd.DataFrame(iris.data, columns=iris['feature_names'])
a=elbow_plot(iris_data[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)']], n_cluster=1)
print (a)
