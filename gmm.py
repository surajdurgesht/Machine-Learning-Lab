import numpy as np
import matplotlib.pyplot as plt
mean = [3, 3]
mean2=[-1,-1]
mean3=[-2,2]
cov=[[1 ,0.7],[0.3,1]]
cov2=[[1 ,0.5],[0.5,1]]
cov3=[[1 ,0.6],[0.4,1]]
x, y = np.random.multivariate_normal(mean, cov, 100).T
x1, y1 = np.random.multivariate_normal(mean2, cov2, 100).T
x2, y2 = np.random.multivariate_normal(mean3, cov3, 100).T
plt.plot(x, y, 'x')
plt.plot(x1,y1,'x')
plt.plot(x2,y2,'x')
plt.axis('equal')
plt.show()
