import numpy as np
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
import scipy as sp
from scipy.stats import multivariate_normal
import os

num = 20
w0_true = -0.3
w1_true = 0.5
alpha = 2
beta = 25
mu = 0
sigma = 0.2
DPI = 200

Xs = np.random.uniform(-1,1,size=num)
y = [w0_true + w1_true*x for x in Xs]
y_noise = y + np.random.normal(mu, sigma, 20)

m1 = np.array([0,0])
s1 = alpha*np.eye(2)

xres = 100
yres = 100

x1 = np.linspace(xlim[0], xlim[1], xres)
y1 = np.linspace(ylim[0], ylim[1], yres)
xx, yy = np.meshgrid(x1,y1)


xxyy = np.c_[xx.ravel(), yy.ravel()]


k1 = sp.stats.multivariate_normal(mean= m1, cov= s1)
posterior = k1.pdf(xxyy)

print()
s1 = np.eye(2)*alpha
m1  = np.zeros((2,1))

img = posterior.reshape((xres,yres))
fig = plt.figure(figsize= (4,4))
c= plt.imshow(img, cmap = 'jet')

ax = plt.gca()

ax.set_xticks([])
ax.set_yticks([])

plt.xlabel(r'$w_1$')
plt.ylabel(r'$w_0$')
plt.title('Heatmap of w_0 and w_1 (Both from -1 to 1)')
fig.colorbar(c, ax=ax)
plt.show()

# for i in range(num):
# 	w1, w0 = np.random.multivariate_normal(mean=m1, cov= s1)
# 	plt.plot(Xs[i], w0 + Xs[i]*w1)
# plt.show()




