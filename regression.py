# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 10:08:45 2019

@author: Suraj
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 19:30:49 2019

@author: Suraj
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm

data = pd.read_csv("D:\Programming\ML\dataset1.csv")

plt.figure(figsize=(16, 8))

plt.scatter(
    data['x'],
    data['y'],
    c='black'
)

plt.show()

X = data['x'].values.reshape(-1,1)
y = data['y'].values.reshape(-1,1)
reg = LinearRegression()
reg.fit(X, y)
print("The linear model is: Y = {:.5} + {:.5}X".format(reg.intercept_[0], reg.coef_[0][0]))

predictions = reg.predict(X)
plt.figure(figsize=(16, 8))
plt.scatter(
    data['x'],
    data['y'],
    c='black'
)
plt.plot(
    data['x'],
    predictions,
    c='blue',
    linewidth=2
)
plt.show()