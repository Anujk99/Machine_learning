# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 10:46:42 2020

@author: HP
"""

###################################################
#linear regression  
import matplotlib.pyplot as plt
xa=[1,2,3,4,5]
ya=[3,4,2,4,5]
plt.scatter(xa,ya,color="red")
x_p=[1,2,3,4,5]
y_p=[2.8,3.2,3.6,4.0,4.4]
plt.plot(x_p,y_p, color="blue")
plt.show()
# calculation of r^2 (coefficient of determination)
from sklearn.metrics import r2_score
r2_score(ya,y_p)


###################################################
# creation of model 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
