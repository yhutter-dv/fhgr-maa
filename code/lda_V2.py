import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import datasets

digits = datasets.load_digits()
x = digits.data
y_true = digits.target

model2 = LDA(n_components=2)
proj2 = model2.fit_transform(x, y_true)
y_pred2 = model2.predict(x)

plt.scatter(proj2[:,0], proj2[:,1], c=y_pred2, \
  cmap=plt.get_cmap('nipy_spectral', 10), alpha=0.5)
plt.colorbar()
plt.show()

model3 = LDA(n_components=3)
proj3 = model3.fit_transform(x, y_true)
y_pred3 = model3.predict(x)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(proj3[:,0], proj3[:,1], proj3[:,2], c=y_pred3, \
  cmap=plt.get_cmap('nipy_spectral', 10), alpha=0.5)
plt.show()

