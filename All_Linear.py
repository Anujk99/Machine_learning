# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 09:42:40 2020

@author: HP
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv(r"D:\churn_data.csv")
data.info()


data=data.drop(["state","account length","area code","phone number"],axis=1)
data.info()



#############################################
#label encoding using loop
#this will convert all object to int
from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()

for i in data:
    if(str(data[i].dtype)=="object"):
        data[i]=label_encoder.fit_transform(data[i])
data.info()
 data["churn"]=label_encoder.fit_transform(data["churn"])




#############################################
#dependent and independent data classification
y=data["churn"]
x=data.drop(["churn"],axis =1 )


##############################################
#split in test and training data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.33, random_state=42)
x_train

#############################################
#linear regression


from sklearn.linear_model import LinearRegression

model=LinearRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
#summary 
####################################
from sklearn.metrics import r2_score


r2_score(y_test,y_pred)



#############################################
#Decision Tree

from sklearn.tree import DecisionTreeRegressor
#model=DecisionTreeClassifier()
model=DecisionTreeRegressor()
dtree=model.fit(x_train,y_train)

##############################################
#prediction



y_pred=model.predict(x_test)



#############################################
#Accuracy check

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

print("classification report is : \n",classification_report(y_test,y_pred))


print("Accuracy of model  is : \n",accuracy_score(y_test,y_pred))

print("Confusion matrix is  : \n",confusion_matrix(y_test,y_pred))


#############################################
####################################

#############################################
#Logistic regression

from sklearn.linear_model import LogisticRegression
#model=DecisionTreeClassifier()
model=LogisticRegression()
dtree=model.fit(x_train,y_train)

##############################################
#prediction



y_pred=model.predict(x_test)


#############################################
#Accuracy check

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

print("classification report is : \n",classification_report(y_test,y_pred))


print("Accuracy of model  is : \n",accuracy_score(y_test,y_pred))

print("Confusion matrix is  : \n",confusion_matrix(y_test,y_pred))
