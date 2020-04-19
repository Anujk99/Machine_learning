# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 11:12:47 2020

@author: HP
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 11:11:03 2020

@author: HP
"""

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
#Applyong Support Vector Mchine 
#SVC : Support Vector classifier
from sklearn.naive_bayes import GaussianNB

model=GaussianNB()
model.fit(x,y)


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




    
    
