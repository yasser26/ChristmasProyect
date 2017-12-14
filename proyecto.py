# -*- coding: utf-8 -*-

# Importación de bibliotecas a usar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

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




def main():
    pathDataSet = raw_input("Ingrese la dirección del set de datos: ")
    mink = int(raw_input("Ingrese el valor del k mínimo: "))
    maxk = int( raw_input("Ingrese el valor del k máximo: "))
    itera_kmeans = int(raw_input("Ingrese la cantidad de veces que iterará el algoritmo kmeas: "))

    data0 = leerDataSet(pathDataSet)
    iris_data1 = obtenerData(data0)
    iris_label1 = obtenerLabels(data0)

    matriz_resultados = []

    for i in range(mink, maxk+1):

        filas_ma_resultados = []
        filas_ma_resultados.append(i)

        kmeans = KMeans(n_clusters = i, max_iter = itera_kmeans).fit(iris_data1)
        filas_ma_resultados.append(metodo_Codo(kmeans))

        filas_ma_resultados.append(metodo_Silueta(kmeans, iris_data1))

        matriz_resultados.append(filas_ma_resultados)


    #print iris_data1
    #print iris_label1
    print matriz_resultados


if __name__ == '__main__':
    main()
