# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 10:57:14 2020

@author: HP
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 09:42:40 2020

@author: HP
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv(r"D:\Mall_Customers.csv")
data.info()


data=data.drop(["CustomerID"],axis=1)
data=data.drop(["Age"],axis=1)
#data=data.drop(["Gender"],axis=1)
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


#############################################
################################################
#extract np array from dataframe
x=data.iloc[0:,0:].values


###############################################
#Applying KMeans and find the value of n_cluster for minimizing wcss

from sklearn.cluster import KMeans
wcss=[]
for i in range(1,12):
    kmean=KMeans(n_clusters=i)
    kmean.fit_predict(x)
    wcss.append(kmean.inertia_)
plt.plot(range(1,12),wcss)
plt.xlabel("Cluster values")

plt.ylabel("WCSS")
plt.title("CUSTOMER VS WCSS")
plt.show()
####################################################
#as per plot values of wcss reduce slowly after a fix point for now 5 so take it as n_cluster
model_final=KMeans(n_clusters=5)
print(model_final)
y_pred=model_final.fit_predict(x)

######################################################
result=pd.DataFrame({"Prediction":y_pred})
data1=pd.read_csv(r"D:\Mall_Customers.csv")
x=data1.iloc[0:,0:]
final_val=pd.concat((x,result),axis=1)

final_val.to_csv(r"D:\Mall_Customers_result.csv")







