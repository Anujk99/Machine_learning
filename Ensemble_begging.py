# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 11:26:24 2020

@author: HP
"""

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

from sklearn.ensemble import BaggingClassifier
model_begging=BaggingClassifier()
model_begging.fit(x_train,y_train)

y_pred=model_begging.predict(x_test)
print(model_begging.score(x,y))
print(y_pred)
print(data.target_names[y_pred])


