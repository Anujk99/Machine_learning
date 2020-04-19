# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 10:18:49 2020

@author: HP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
data=load_iris()


print(data)
print(data.target)
print(data.data)


######################################
#assigning data to x and y that is dependent and independent
y=data.target

x=data.data


##########################################
#Applyong KNN 
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=5)
model.fit(x,y)

#custom test case
"""x_pred=[[5.1,3.5,1.4,0.2],
[4.9	,3,	1.4,	0.2],
[4.7,	3.2	,1.3	,0.2],
[4.6,	3.1,	1.5	,0.2],
[5	,3.6,	1.4	,0.2],
[5.4,	3.9,	1.7	,0.4
]]

y_pred=model.predict(x_pred)  
print("*"*50)
for i in y_pred:
    print("Prediction is on :",data.target_names[i])
"""

##############################################
    
#using user define input
def show(x_test):
    y_pred=model.predict([x_test,])  
    print("*"*50)
    print("Prediction is on :",data.target_names[y_pred])
    

sl=float(input("Enter Sepal Length"))
sw=float(input("Enter Sepal Width"))
pl=float(input("Enter Petal Length"))
pw=float(input("Enter Petal Width"))
    
test=[sl,sw,pl,pw]

show(test)
##############################################

    
    
