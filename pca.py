# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 09:42:54 2020

@author: HP
"""


import pandas as pd
import numpy as np

data=pd.read_csv(r"D:\Wine.csv")
print(data.info())


x=data.drop(["Customer_Segment"],axis=1).values
y=data["Customer_Segment"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=42)
 
 
 ###############################################
 #scale
 
from sklearn.preprocessing import StandardScaler

# Standardizing the features
x_train = StandardScaler().fit_transform(x_train)

x_test = StandardScaler().fit_transform(x_test)

from sklearn.decomposition import PCA

pca=PCA(n_components= 3)
x_train=pca.fit_transform(x_train)
x_test=pca.fit_transform(x_test)




opt_com=pca.explained_variance_

opt_com_ratio=pca.explained_variance_ratio_

######################################################

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)



from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

print("classification report is : \n",classification_report(y_test,y_pred))

print("Accuracy of model  is : \n",accuracy_score(y_test,y_pred))

print("Confusion matrix is  : \n",confusion_matrix(y_test,y_pred))



















