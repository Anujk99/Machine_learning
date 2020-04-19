

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv(r"E:\PY\Machine-Learning-Tutorial-master\Machine-Learning-Tutorial-master\data\id3.csv")
data.info()

for i in data:
    print(data)
    
P=data.iloc[0:,0:]
j=14
for i in range(0,len(data)):
    if(data.iloc[i,1]!='hot'):
        print((data.iloc[i:i+1,0:]))
        
    

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
y=data["Answer"]
x=data.drop(["Answer"],axis =1 )




##############################################
#split in test and training data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.33, random_state=4)
x_train


#############################################
#Decision Tree

from sklearn.naive_bayes import GaussianNB

model=GaussianNB()
model.fit(x_train,y_train)


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
