from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt("data/decomp_test.dat", delimiter=",")
model = PCA(n_components=2)

data_proj = model.fit_transform(data)
y = np.zeros([len(data_proj)])
plt.plot(data_proj[:,0], y, 'mo')
plt.plot(y, data_proj[:,1], 'co')
ax = plt.gca()
ax.set_aspect(1)
plt.show()