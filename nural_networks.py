# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 21:16:39 2019

@author: Suraj
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn.datasets import fetch_mldata, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score

from prml import nn
#import math as m

np.random.seed(1234)

class RegressionNetwork(nn.Network):
    
    def __init__(self, n_input, n_hidden, n_output):
        truncnorm = st.truncnorm(a=-2, b=2, scale=1)
        super().__init__(
            w1=truncnorm.rvs((n_input, n_hidden)),
            b1=np.zeros(n_hidden),
            w2=truncnorm.rvs((n_hidden, n_output)),
            b2=np.zeros(n_output)
        )

    def __call__(self, x, y=None):
        h = nn.tanh(x @ self.w1 + self.b1)
        self.py = nn.random.Gaussian(h @ self.w2 + self.b2, std=1., data=y)
        return self.py.mu.value

def create_toy_data(func, n=50):
    x = np.linspace(-1, 1, n)[:, None]
    return x, func(x)

def sinusoidal(x):
    return np.sin(np.pi * x)

def heaviside(x):
    return 0.5 * (np.sign(x) + 1)


func_list = [np.square, sinusoidal, np.abs, heaviside]
plt.figure(figsize=(14, 6))
x = np.linspace(-1, 1, 1000)[:, None]
for i, func, n_iter, decay_step in zip(range(1, 5), func_list, [1000, 10000, 10000, 10000], [100, 100, 1000, 1000]):
    plt.subplot(2, 2, i)
    x_train, y_train = create_toy_data(func)
    model = RegressionNetwork(1, 3, 1)
    optimizer = nn.optimizer.Adam(model, 0.1)
    optimizer.set_decay(0.9, decay_step)
    for _ in range(n_iter):
        model.clear()
        model(x_train, y_train)
        log_likelihood = model.log_pdf()
        log_likelihood.backward()
        optimizer.update()
    y = model(x)
    plt.scatter(x_train, y_train, s=10)
    plt.plot(x, y, color="r")
plt.show()