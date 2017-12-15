# -*- coding: utf-8 -*-

# Importación de bibliotecas a usar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import pairwise_distances
from sklearn import metrics


# Función que lee el set de datos. Recibe como parámetro el path del dataset
def leerDataSet(pathDataSet):
    data0 = pd.read_csv(pathDataSet);
    return data0;

# Función que obtiene los datos del dataset
def obtenerData(data0):
    data = np.asarray(data0);
    tags = np.asarray(data0.columns.values);

    iris_data = pd.DataFrame(data[:data.shape[0],:data.shape[1]-1], columns=tags[:data.shape[1]-1]);
    return iris_data;

# Función que obtiene los labels del dataset
def obtenerLabels(data0):
    data = np.asarray(data0);
    tags = np.asarray(data0.columns.values);

    ff = data[:data.shape[0],-1];
    ff2 = ff.reshape(data.shape[0], 1);

    iris_label = pd.DataFrame(ff2, columns=[tags[data.shape[1]-1]]);
    return iris_label;

# Función que ejecuta el metodo del codo. Retorna la suma de distancias al cuadrado
# de las muestras a su centro de clúster más cercano.
def metodo_Codo(kmeans):
    valor = kmeans.inertia_;
    return valor;

# Función que ejecuta el metodo de la silueta.
def metodo_Silueta(kmeans, iris_data):
    avg = silhouette_score(iris_data, kmeans.labels_);
    return avg;

# Función que retorna del indice Calinski-Harabaz
def indice_Calinski_Harabaz(kmeans, iris_data):
    index = metrics.calinski_harabaz_score(iris_data, kmeans.labels_)
    return index

def metodo_Gap(kmeans, iris_data, i):

    refDisps = np.zeros(3);
    for u in range(3):

        # Create new random reference set
        randomReference0 = np.random.random_sample(size=iris_data.shape);
        randomReference = pd.DataFrame(randomReference0);
        # Fit to it
        km = KMeans(n_clusters = i).fit(randomReference);

        refDisp = km.inertia_;
        refDisps[u] = refDisp;

    gap = np.log(np.mean(refDisps)) - np.log(kmeans.inertia_);
    return gap;





def main():
    #pathDataSet = raw_input("Ingrese la dirección del set de datos: ")
    mink = int(raw_input("Ingrese el valor del k mínimo: "))
    maxk = int( raw_input("Ingrese el valor del k máximo: "))
    itera_kmeans = int(raw_input("Ingrese la cantidad de veces que iterará el algoritmo kmeas: "))

    data0 = leerDataSet(pathDataSet)
    iris_data1 = obtenerData(data0)
    iris_label1 = obtenerLabels(data0)


    matriz_resultados0 = []

    for i in range(mink, maxk+1):

        filas_ma_resultados = []
        filas_ma_resultados.append(i)

        kmeans = KMeans(n_clusters = i, max_iter = itera_kmeans).fit(iris_data1)

        filas_ma_resultados.append(metodo_Codo(kmeans))
        if(i!=1):
            filas_ma_resultados.append(metodo_Silueta(kmeans, iris_data1))
        else:
            filas_ma_resultados.append(0)

        if(i!=1):
            filas_ma_resultados.append(indice_Calinski_Harabaz(kmeans, iris_data1))
        else:
            filas_ma_resultados.append(0)


        filas_ma_resultados.append(metodo_Gap(kmeans, iris_data1, i))

        matriz_resultados0.append(filas_ma_resultados)

    matriz_resultados = np.asarray(matriz_resultados0)
    titles = ['k','Elbow Method','Silhouette','Calinski-Harabaz', 'Gap Method']
    for q in range(1,5):

        plt.plot(list(matriz_resultados[:,0]), list(matriz_resultados[:, q]), marker='x', linestyle=':', color='b', label = titles[q])
        plt.show()


    impri_ma = pd.DataFrame(matriz_resultados0, columns=titles)
    impri_ma.to_csv('prueba.csv', sep='\t', index=False)


if __name__ == '__main__':
    main()
