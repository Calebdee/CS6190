import numpy as np
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt

w0_true = -0.3
w1_true = 0.5
Xs = np.random.uniform(-1,1,size=20)

preds = [(w0_true + w1_true*x) + np.random.normal(loc=0, scale=0.2, size=1)[0] for x in Xs]
print(preds)

prior_data = [np.random.normal(0, 2, 2) for x in range(20)]

i=0
for prior in prior_data:
	prior[1] *= Xs[i]
	i +=1


ax = sns.heatmap(prior_data, vmin=-1, vmax=1)
def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')

i=0
for prior in prior_data:
	abline(prior[1]*Xs[i], prior[0])
	i += 1

plt.title("Lines of w prior draws")
plt.show()



