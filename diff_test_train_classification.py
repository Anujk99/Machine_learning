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

################################
#training
x_train=data.drop(["Survived"],axis =1 )
y_train=data["Survived"]

######################## 
# applying logistic regresiom
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)


################################################
#                   for testing

data1=pd.read_csv(r"D:\test.csv")
data1.info()

data1=data1.drop(["PassengerId", "Name", "Ticket","Cabin"], axis=1)
data1.info()

###############################
#   DATA CLEANING

data1["Age"].hist()
data1["Age"].fillna(data["Age"].median(),inplace=True)
data1["Fare"].hist()
data1["Fare"].fillna(data["Fare"].median(),inplace=True)


########################
data1.isnull().sum()
data1.info()

#label encoding
from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
data1["Sex"]=label_encoder.fit_transform(data1["Sex"])

data1["Embarked"]=label_encoder.fit_transform(data1["Embarked"])
data1.info()
x_test=data1
#now prediction

y_pred=model.predict(x_test)
###################
#total saved person
print(y_pred.sum())


############################setting threshold values

THRESHOLD=.50
y_pred=np.where(model.predict_proba(x_test)[:,1]>THRESHOLD,1,0)

print(y_pred.sum())
