# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 09:38:34 2020

@author: HP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv(r"D:\ibm_hr.csv")
data.info()


data=data.drop(["EmployeeCount"],axis=1)
data=data.drop(["EmployeeNumber"],axis=1)
data=data.drop(["Over18"],axis=1)


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
#dependent and independent data classification
y=data["Attrition"]
x=data.drop(["Attrition"],axis =1 )


##############################################
#split in test and training data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.33, random_state=42)
x_train


#############################################
#Decision Tree

from sklearn.tree import DecisionTreeClassifier
#model=DecisionTreeClassifier()
model=DecisionTreeClassifier(criterion="entropy")
dtree=model.fit(x_train,y_train)

##############################################
#prediction

THRESHOLD=.40
y_pred=np.where(model.predict_proba(x_test)[:,1]>THRESHOLD,1,0)
#y_pred=model.predict(x_test)




############################################
#drawing of Decision tree
import pydotplus

import matplotlib.image as pltimg

from sklearn import tree





data1 = tree.export_graphviz(dtree, out_file=None, feature_names=x_test.iloc[0])
graph = pydotplus.graph_from_dot_data(data1)


graph.write_png('D:\mydecisiontree.png')

img=pltimg.imread('D:\mydecisiontree.png')
imgplot = plt.imshow(img)
plt.show()








#############################################
#Accuracy check




from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

print("classification report is : \n",classification_report(y_test,y_pred))


print("Accuracy of model  is : \n",accuracy_score(y_test,y_pred))

print("Confusion matrix is  : \n",confusion_matrix(y_test,y_pred))
