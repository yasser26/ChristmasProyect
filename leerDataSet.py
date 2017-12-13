import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.cluster import KMeans



def main():
    data = pd.read_csv('iris_dataset.csv')
    print data




if __name__ == '__main__':
    main()
