# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 10:33:13 2020

@author: HP
"""
###############################
#       LOGISTIC REGRESSION
##############################


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data=pd.read_csv(r"D:\churn_data.csv")
data.info()



###############################
#   DATA CLEANING
##############################
data.shape
data.columns
data.info()
########################
data.isnull().sum()

###############################
#
##############################
#removeed some irrelevent data
data_pre=data.iloc[0:, 4:]
data_pre.info()

#dummies label encoding
ip=pd.get_dummies(data_pre["international plan"])
ip.head()
ip.pop("no")

vp=pd.get_dummies(data_pre["voice mail plan"])
vp.head()
vp.pop("yes")
vp.head()

chu_red=pd.get_dummies(data_pre["churn"], drop_first=True)
chu_red.head()
##########################
data_p=pd.concat((data_pre, ip, vp, chu_red),axis=1)
data_p.shape

final=data_p.drop(["churn", "voice mail plan", "international plan"], axis=1)
final.shape
final.info()

###################
#dividing into dep and independent 
y=final[True]
x=final.drop([True], axis=1)



###################################
# train teat split

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=21)
x_train
######################## 
# applying logistic regresiom
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





