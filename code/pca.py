from sklearn import datasets
import matplotlib.pyplot as plt
data, shape = datasets.make_swiss_roll(n_samples=1000, noise=0.0)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(data[:,0], data[:,1], data[:,2], c=shape)
plt.show()