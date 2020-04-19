# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 10:29:25 2020

@author: HP
"""

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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=11)
x_train
######################## 
# applying logistic regresiom
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()

model.fit(x_train,y_train)

y_pred=model.predict(x_test)

from sklearn.metrics import accuracy_score



from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

print("classification report is : \n",classification_report(y_test,y_pred))


print("Accuracy of model  is : \n",accuracy_score(y_test,y_pred))

print("Confusion matrix is  : \n",confusion_matrix(y_test,y_pred))

##################################
p=0
max=0;
t=.01
t1=t
while t<1.0:    
    THRESHOLD=t
    y_pred=np.where(model.predict_proba(x_test)[:,1]>THRESHOLD,1,0)
    t=t+.01
    p=accuracy_score(y_test,y_pred)
    if p > max:
        max=p
        t1=t
    
print(max)
print(t1 )




