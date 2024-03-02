import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score as score

data = np.loadtxt('./data/credit.dat', delimiter=',')
x_train, x_test, y_train, y_test = tts(data[:,:2], data[:,2], test_size=0.3)
model = knn(n_neighbors=7)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(score(y_test, y_pred))

col_train = ListedColormap(['red', 'blue'])
col_test = ListedColormap(['orange', 'cyan'])

# c = Predefined classification
sp = plt.scatter(x_train[:,0], x_train[:,1], c=y_train, cmap=col_train)
plt.scatter(x_test[:,0], x_test[:,1], c=y_test, cmap=col_test)

lab = ['niedrig', 'hoch']
plt.legend(handles=sp.legend_elements()[0], labels=lab)
plt.grid(True)
plt.show()
