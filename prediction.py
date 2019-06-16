import numpy as np 
from numpy.linalg import inv
import matplotlib.pyplot as plt

model = 10
sigma = 0.3
lambada = 0.001

alpha = 0.1
beta = 9.0

xmin = -1
xmax = 1

def true_fun(x):
    y = np.sin(2*np.pi*x)
    #y = x**2
    return y

def phi_polynomial(x , i):

    return np.power(x,i)
    

def phi_gussian(x , i):
    
    if(i==0):
         return np.power(x,0)
    mu = np.min(x) + ((np.max(x) - np.min(x))/model)*i
   
    return (np.exp(-(x - mu)**2/ ( 2*(sigma**2) ) ) )

def phi_sigmoid(x , i):
    
    if(i==0):
         return np.power(x,0)
    mu = np.min(x) + ((np.max(x) - np.min(x))/model)*i
   
    a = -(x - mu)/ sigma
    return (np.exp(a) / (1+np.exp(a) ))


def phi(x,i):
    return phi_polynomial(x,i)


def design_matrix(x,degree):   
    Q = np.empty([x.shape[0], 0])
    phi(x,1)
    for i in range(degree+1):
        Q = np.concatenate((Q,phi(x,i) ),axis=1)
    return Q   

def feature(x,degree):

    Q = np.empty([x.shape[0], 0])
    phi(x,1)
    for i in range(degree+1):
        Q = np.concatenate((Q,phi(x,i) ),axis=1)
    return Q
    
def bayesian(Q,t):
    S = inv(alpha * np.identity(Q.shape[1]) + beta * np.dot(Q.T, Q))
    mu = beta * np.dot(S, np.dot(Q.T, t))
    v = 1/beta + np.dot( np.dot(Q, S) , Q.T)
    variance = np.diagonal(v)
    
    sd = np.power(variance , 0.5)
    return mu,sd

x = np.linspace(xmin,xmax,100)

y = true_fun(x)


train_x = np.linspace(xmin, xmax, 20)
t = true_fun(train_x)+ np.random.normal(scale=0.25, size=train_x.shape)

t = t.T

train_x.resize(len(train_x),1)

Q = design_matrix(train_x,model)


w = np.dot(np.dot(inv(np.dot(Q.T ,Q) + lambada*np.identity(model+1) ),Q.T ) , t)

x_t = np.linspace(-1,1,100)
x_t.resize(len(x_t),1)
x_t_feature = design_matrix(x_t,model) 
y_t = x_t_feature.dot(w.T)

mu_n  , s_n = bayesian(x_t_feature,y_t)

y_bayesian = x_t_feature.dot((mu_n).T)


y_1 = y_bayesian + s_n

y_2 =y_bayesian -s_n

print(y_1.shape)
y_1.resize(len(x_t),1)
y_2.resize(len(x_t),1)


print(y_1.shape)




plt.fill_between(x_t.flatten(), y_2.flatten(), y_1.flatten(), color='pink', alpha='0.5')

plt.plot(x_t,y_t,c="r")

plt.scatter(train_x,t)
plt.plot(x,y,c="g")

np.set_printoptions(suppress=True)


plt.show()          
