# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 09:36:54 2020

@author: HP
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data=pd.read_csv(r"D:\Social_Network_Ads.csv")
data.info()

#remove unwanted data
#removeed some irrelevent data

data=data.drop(["User ID"], axis=1)
data.info()

###############################
#   DATA CLEANING
##############################
data["Age"].hist()
data["Age"].fillna(data["Age"].mean(),inplace=True)

########################
data.isnull().sum()

data["EstimatedSalary"].hist()
data["EstimatedSalary"].fillna(data["EstimatedSalary"].median(),inplace=True)
data.isnull().sum()

data.info()
#label encoding
from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
data["Gender"]=label_encoder.fit_transform(data["Gender"])
#here we have two missing values so we have to fill them for label encoding 

data.info()

x=data.drop(["Purchased"],axis =1 )

y=data["Purchased"]

###################################
# train teat split

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=10)
x_train
######################## 
#applying KNN 

from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=5)
model.fit(x_train,y_train)

THRESHOLD=.50
y_pred=np.where(model.predict_proba(x_test)[:,1]>THRESHOLD,1,0)

print(y_pred)

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

print("classification report is : \n",classification_report(y_test,y_pred))


print("Accuracy of model  is : \n",accuracy_score(y_test,y_pred))

print("Confusion matrix is  : \n",confusion_matrix(y_test,y_pred))



