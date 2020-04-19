# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 11:40:58 2020

@author: HP
"""



# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 10:03:30 2020

@author: HP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv(r"D:\Market_Basket_Optimisation.csv",header=None)
data.info()


data.shape
record=[]

temp=[]

def ch(temp,p):
    if p in temp:
        return False
    else:
        return True

for i in data:
    for j in range(0,7500):
        p=data[i][j]
        if(ch(temp,p)):
            temp.append(p)
            
            
        
for i in range(0,7501):
    record.append([str(data.values[i,j])
    for j in range(0,20)])
    
    

    
from apyori import apriori
rule=apriori(record,min_support=.04,min_confidence=0.02,min_lift=3,min_length=2)
print(rule)







