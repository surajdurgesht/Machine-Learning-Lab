# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 10:49:43 2019

@author: Suraj
"""
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd


def step_function(x):
    if x<0:
        return 0
    else:
        return 1

training_set = [((0, 0), 0),((1, 1), 1)]

x1 = [training_set[i][0][0] for i in range(2)]
x2 = [training_set[i][0][1] for i in range(2)]
y = [training_set[i][1] for i in range(2)]
print(x1)
print(x2)
print(y)

df = pd.DataFrame(
    {'x1': x1,
     'x2': x2,
     'y': y
    })
    
sns.lmplot("x1", "x2", data=df, hue='y', fit_reg=False, markers=["o", "s"])

w = np.random.rand(2)
print(w)
errors = [] 
eta = .5
epoch = 2000
b = 0

for i in range(epoch):
    for x, y in training_set:
        u = sum(x*w) + b
      
        error = y - step_function(u) 
      
        errors.append(error) 
        for index, value in enumerate(x):
            w[index] += eta * error * value
            b += eta*error

a = [0,-b/w[1]]
c = [-b/w[0],0]
plt.plot(a,c)
plt.show()
