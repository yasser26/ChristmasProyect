# -*- coding: utf-8 -*-

#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.cluster import KMeans




def leerDataSet():
    data0 = pd.read_csv('iris_dataset.csv');
    tags = np.asarray(data0.columns.values);
    data = np.asarray(data0);
    #print data;

    ff = data[:data.shape[0],-1];
    ff2 = ff.reshape(data.shape[0], 1);
    #print ff2;
    #print data.shape;

    iris_data = pd.DataFrame(data[:data.shape[0],:data.shape[1]-1], columns=tags[:data.shape[1]-1]);
    iris_label = pd.DataFrame(ff2, columns=[tags[data.shape[1]-1]]);
    print iris_data;
    print iris_label;


def main():
    leerDataSet()


if __name__ == '__main__':
    main()
