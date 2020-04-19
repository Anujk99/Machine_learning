# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 09:43:26 2020

@author: HP
"""

import matplotlib.pyplot as plt
xa=[1,2,3,4,5]
ya=[3,4,2,4,5]
plt.scatter(xa,ya,color="red")

x_p=[1,2,3,4,5]
y_p=[2.8,3.2,3.6,4.0,4.4]
plt.plot(x_p,y_p, color="blue")
plt.show()
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#from sklearn.linear_model import LinearRegression
#from sklearn.metrics import accuracy_score

from sklearn.metrics import r2_score
r2_score(ya, y_p)

################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import data
data=pd.read_csv(r"E:\PY\data\Regression.csv")
data.info()
#####################
data.isnull().sum()
data["Age"].plot.hist()
data["Age"].fillna(data["Age"].median(),inplace=True)

data.info()
#convert to dummies
jt=pd.get_dummies(data["Job Type"],drop_first=True)
jt.head()

Ms=pd.get_dummies(data["Marital Status"],drop_first=True)
Ms.head()


Edu=pd.get_dummies(data["Education"])#,drop_first=True)
Edu.pop("Secondry")
Edu.head()


city=pd.get_dummies(data["Metro City"])#,drop_first=True)
city.pop("Yes")
city.head()
####################
#concat all dummies to my data
pre_final=pd.concat((data,jt,Ms,Edu,city),axis=1)

#data drop
final_data=pre_final.drop(["Metro City","Education","Marital Status","Job Type"],axis=1)

#outlier treatment
import seaborn as sns
sns.boxplot(final_data["Age"])

final_data["Age"]=np.where(final_data["Age"]>60,60,final_data["Age"])
#treated box plt
sns.boxplot(final_data["Age"])
#############################################
# sep dependent and independent
y=final_data["Purchase made"]
x=final_data.drop(["Purchase made"],axis=1)
#######################################################3
#data split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
x_train.shape
y_train.shape
x_test.shape
x_test.to_csv(r"C:\Users\fluXcapacit0r\Desktop\test.csv")
x_t=pd.read_csv(r"C:\Users\fluXcapacit0r\Desktop\test.csv")
####################################################
#model
from sklearn.linear_model import LinearRegression
model=LinearRegression()

#model train
model.fit(x_train,y_train)

#predict value of purchase made
y_pred=model.predict(x_t)
#summary 
####################################
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
