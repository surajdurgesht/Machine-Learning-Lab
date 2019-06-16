# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 15:49:43 2019

@author: Suraj
"""

from random import *
from math import sqrt

inside = 0
n = 1000

for i in range(0,n):
    x = random()
    y = random()
    
    if sqrt(x*x+y*y) <= 1:
        inside += 1

pi = 4*inside/n
print(pi)        
