# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 10:03:30 2020

@author: HP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

data=pd.read_csv(r"D:\Mall_Customers.csv")
x=data.iloc[0:,[3,4]]
dendo=sch.dendrogram(sch.linkage(x,method='ward')) 
#ward minimizes the Varience within cluster and KMeans++ minimize WCSS
plt.title('Dendrogram')
plt.xlabel('CUstomers')
plt.ylabel("ED")
plt.show()

from sklearn.cluster import hierarchical
model=hierarchical.AgglomerativeClustering(n_clusters=4 )
y_sch=model.fit_predict(x)

