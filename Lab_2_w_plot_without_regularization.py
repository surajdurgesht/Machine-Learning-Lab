"""
Created on Mon Jan 21 12:58:53 2019

@author: Arun Chauhan
"""
import numpy as np 
from numpy.linalg import inv
import matplotlib.pyplot as plt


model = 9

'''True Funtion'''
x_true = np.linspace(0,1,100)
y_true = np.sin(2*np.pi*x_true)

'''Polynomial Feature'''
def feature(x,degree):   
    X = np.empty([x.shape[0], 0])
    for i in range(degree+1):
        X = np.concatenate((X,np.power(x,i)),axis=1)
    return X    
    

x = np.linspace(0, 1, 10)
y = np.sin(2*np.pi*x) 
t = y + np.random.normal(scale=0.25, size=x.shape)
t = t.T

x.resize(len(x),1)
X = feature(x,model)

w = inv(X.T.dot(X)).dot(X.T).dot(t)  


x_t = np.linspace(0,1,100)
x_t.resize(len(x_t),1)
x_t_feature = feature(x_t,model) 
y_t = x_t_feature.dot(w.T)
plt.plot(x_t,y_t,c="r")
plt.scatter(x,t)
plt.plot(x_true,y_true,c="g")



np.set_printoptions(suppress=True)
print(w) 

#plt.savefig('plot_deg_0.png')          
