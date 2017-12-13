from __future__ import print_function
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
print(__doc__)
## Recibe los datos y el valor de k en ese momento, retorna el promedio
def Metodo_silueta(datos, n_clusters):
	## Debemos obtener datos "X" dentro del programa principal y mandarselo a esta funcion
	## Recordar que no funciona con k 1, debemos crear dos condiciones 
	## para poder saltarnos este metodo en ese momento
	clusterer = KMeans(n_clusters=n_clusters, random_state=10)
	cluster_labels = clusterer.fit_predict(datos)
	avg = silhouette_score(datos, cluster_labels)
	return avg	
X, y = make_blobs(n_samples=500,
					n_features=2,
					centers=4,
					cluster_std=1,
					center_box=(-10.0, 10.0),
					shuffle=True,
					random_state=1)
avg = Metodo_silueta(X, 2)
print("El promedio es: ",avg)
print(X)
