# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 09:29:25 2020

@author: HP
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data=pd.read_csv(r"D:\train.csv")
data.info()

#remove unwanted data
#removeed some irrelevent data

data=data.drop(["PassengerId", "Name", "Ticket","Cabin"], axis=1)
data.info()

###############################
#   DATA CLEANING
##############################
data["Age"].hist()
data["Age"].fillna(data["Age"].median(),inplace=True)

########################
data.isnull().sum()


#label encoding
from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
data["Sex"]=label_encoder.fit_transform(data["Sex"])
#here we have two missing values so we have to fill them for label encoding 
data["Embarked"].fillna(method='ffill',inplace=True)
data["Embarked"]=label_encoder.fit_transform(data["Embarked"])
data.info()

x=data.drop(["Survived"],axis =1 )

y=data["Survived"]

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



##########################################################3
#Apply KNN eith eculidean distance




from sklearn.neighbors import KNeighborsClassifier
model2=KNeighborsClassifier(n_neighbors=7,metric="euclidean")

model2.fit(x_train,y_train)

THRESHOLD=.50
y_pred=np.where(model.predict_proba(x_test)[:,1]>THRESHOLD,1,0)             

print(y_pred)

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

print("classification report is : \n",classification_report(y_test,y_pred))


print("Accuracy of model  is : \n",accuracy_score(y_test,y_pred))

print("Confusion matrix is  : \n",confusion_matrix(y_test,y_pred))













