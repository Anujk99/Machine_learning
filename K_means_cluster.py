# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 10:26:24 2020

@author: HP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
x=np.array([[185,72],[170,56],[168,60],[179,68],[188,77],[180,71],[180,70],[183,74],
            [180,88],[180,67],[177,76]],dtype=int)

from sklearn.cluster import KMeans
model=KMeans(n_clusters=2)
y_pred=model.fit_predict(x)
print(y_pred)
