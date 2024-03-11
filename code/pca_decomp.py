from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt("data/decomp_test.dat", delimiter=",")
model = PCA(n_components=2)
model.fit(data)
print("com:", model.components_)
print(model.explained_variance_ratio_)
plt.plot(data[:,0], data[:,1], 'mo')
plt.quiver([0, 0], [0, 0], model.components_[:,0], model.components_[:,1], color=['b', 'g'], scale=5)
plt.axis([-2.3, 2.3, -2.3, 2.3])
ax = plt.gca()
ax.set_aspect(1)
plt.show()