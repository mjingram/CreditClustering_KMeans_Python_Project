#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 12:04:14 2018

@author: mingram
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from os.path import expanduser
import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.decomposition import PCA
import pylab as pl


## Load Data
hd = expanduser('~')
wd = hd+'/MATH 6338/'
ccDf=pd.read_csv(wd + 'ccDat.csv')
np.random.seed(123)

## Clean Data and Standardize
ccDf.isnull().values.sum()
ccDf.info()
ccDf.describe()
ccDf.drop('Unnamed: 0', axis = 1, inplace =True)
dfStd=preprocessing.scale(ccDf)
dfStd=pd.DataFrame(dfStd, columns=ccDf.columns)

dfStd.info()
dfStd.describe()

## First Determine K Using Elbow plot
distortions=[]
K = range(1,20)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(dfStd)
    kmeanModel.fit(dfStd)
    distortions.append(kmeanModel.inertia_)

plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

## PCA dimension reduction
pca = PCA(n_components=2).fit(dfStd)
pca_2d = pca.transform(dfStd)
pl.scatter(pca_2d[:,0],pca_2d[:,1],c='black')
pl.show()

## k-means
kmeans = KMeans(n_clusters=4)
kmeans.fit(dfStd)
labels = kmeans.predict(dfStd)
centroids = kmeans.cluster_centers_
centroids

## Plot Results

for i in range(0, pca_2d.shape[0]):
    if kmeans.labels_[i] == 1:
        c1 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='r',
                        marker='+')
    elif kmeans.labels_[i] == 0:
        c2 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='g',
                        marker='o')
    elif kmeans.labels_[i] == 2:
        c3 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='b',
                        marker='*')
    elif kmeans.labels_[i] == 3:
        c4 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='y',
                        marker='x')
pl.legend([c1, c2, c3, c4],['Cluster 1', 'Cluster 0',
                  'Cluster 2', 'Cluster 3'])
pl.title('K-means 4 Clusters')
pl.show()

