from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt

k = 3
max_iter = 100
data = np.loadtxt('./data/clicks.dat', delimiter=',')

model = KMeans(n_clusters=k, max_iter=max_iter)
y_pred = model.fit_predict(data)

print(y_pred)

col = ListedColormap(['red', 'blue', 'green'])
sp = plt.scatter(data[:,0], data[:,1], c=y_pred, cmap=col)
lab = (['C1', 'C2', 'C3', 'C4', 'C5'])
plt.legend(handles=sp.legend_elements()[0], labels=lab)
plt.show()

inert = []
for i in range(1,11):
    model = KMeans(n_clusters=i, max_iter=max_iter)
    model.fit(data)
    inert.append(model.inertia_)

x = np.linspace(1,10,10)
plt.plot(x, inert, 'b-')
plt.plot(x, inert, 'bo')
plt.show()