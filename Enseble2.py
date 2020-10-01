# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 11:11:40 2020
@author: HP
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris 

data=load_iris()
x=data.data
y=data.target


###################################
# train teat split

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=29)
x_train

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()

from sklearn.svm import SVC
svm_m=SVC()

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()


from sklearn.ensemble import VotingClassifier
model_en=VotingClassifier([("Decision_tree",dt),("support_vector",svm_m),("KNN",knn)])

model_en.fit(x_train,y_train)
y_pred=model_en.predict(x_test)
print(model_en.score(x,y))
print(y_pred)
print(data.target_names[y_pred])
