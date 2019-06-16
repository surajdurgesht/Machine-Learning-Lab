# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 12:10:58 2019

@author: Suraj
"""
import numpy as np
import matplotlib.pyplot as plt
#
x_true = np.linspace(0,1,100)
y_true = np.sin(2*np.pi*x_true)


x = np.linspace(0,1,10)
#A = np.array([[1,-1,2],[3,2,0]])
mu = 0
sigma = 1

noise =  np.random.normal(scale=0.25, size=x.shape) 
temp = np.sin(2*np.pi*x)
t = temp + noise
#t_t = y_t +noise

#poy = np.polyfit(x, t, 9)
#print(poy)
#poy1 = np.poly1d(poy)
#xpoy = np.linspace(0, max(x), 100)
#ypoy = poy1(xpoy)

a = []
for i in range (0, len(x)):
    a.append(1)
    
 
#deg = 3
#for z in range (0, deg):
#    q = x**deg
#    print(q)
#a = np.array((1,1,1,1,1,1,1,1,1,1))
    
#b = np.column_stack((a, x, x**2, x**3,x**4, x**5, x**7, x**8, x**9))
f = np.column_stack(a)
b = list(f)
deg = 9

for i in range (0,deg):
  b.append(np.power(x,i))

#np.asmatrix(b)
#b = np.reshape(10, 1)

#c = np.transpose(b)
#d = np.matmul(c,b)
#e = np.linalg.inv(d)
#f = np.matmul(e,c) 
#w = np.matmul(f,t)

w = np.matmul(np.matmul((np.linalg.inv(np.matmul(np.transpose(b), b))), np.transpose(b)), t)
print(w)
#print(w)
##w = d * c * t
plt.plot(x_true,y_true,c="b")
plt.scatter(x, t,color = 'red')
plt.plot(x, t,color = 'green')
#plt.plot(x, w, color = 'blue')
#plt.plot(x, temp)
plt.show()
#print(x.T *x)


    

