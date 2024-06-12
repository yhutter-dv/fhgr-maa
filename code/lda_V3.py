import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split as tts
from sklearn import datasets

faces = datasets.fetch_olivetti_faces()

x, x_test, y_true, y_test = tts(faces.data, faces.target, test_size=0.01, random_state=42)

model_lda = LDA(n_components=3)
model_lda.fit(x, y_true)
proj = model_lda.transform(x)
test_proj = model_lda.transform(x_test)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(proj[:,0], proj[:,1], proj[:,2], c=y_true, \
  cmap=plt.get_cmap('nipy_spectral', 40), alpha=0.5)
plt.show()

hits = [0]
for k in range(1,11):
  model_knn = knn(k)
  model_knn.fit(proj, y_true)
  y_pred = model_knn.predict(test_proj)
  hits.append(accuracy_score(y_test, y_pred))

plt.step(np.linspace(0, 10, 11), hits, where='mid', color='red', linewidth=2)
plt.grid(True)
plt.show()



