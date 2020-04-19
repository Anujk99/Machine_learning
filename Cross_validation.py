# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 09:43:09 2020

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



from sklearn.svm import SVC
model=SVC()
model.fit(x_train,y_train)
###############################################

from sklearn.model_selection import cross_val_score

accuracy=cross_val_score(model,x_train,y_train,cv=10)
print(accuracy)


from sklearn.model_selection import GridSearchCV
par=[{"C":[1,10,100,1000],"kernel":["linear"]},
     {"C":[1,10,100,1000],"kernel":["rbf"],"gamma":[0.1,0.01,0.001,.0001]}]


GS=GridSearchCV(estimator=model,param_grid=par,scoring="accuracy",cv=10,n_jobs=-1)


model_gs=GS.fit(x_train,y_train)


print(model_gs.best_score_)

print(model_gs.best_params_)



###############################################
#Final model : optimal for SVC :{'C': 1, 'kernel': 'linear'} ,0.9821428571428571
opt_model=SVC(C=1,kernel="linear")
opt_model.fit(x_train,y_train)
y_pred=opt_model.predict(x_test)
print(y_pred)
print(data.target_names[y_pred])

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

print("classification report is : \n",classification_report(y_test,y_pred))

print("Accuracy of model  is : \n",accuracy_score(y_test,y_pred))

print("Confusion matrix is  : \n",confusion_matrix(y_test,y_pred))