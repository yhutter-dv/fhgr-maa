from tensorly.decomposition import tucker
from sklearn.datasets import fetch_olivetti_faces
import tensorly as tl
from plot import plot_faces
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split as tts
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.metrics import accuracy_score
import numpy as np

faces = fetch_olivetti_faces()
G, fac = tucker(faces.data, (32,32,32))
#plot_faces(G, 4, 8, rnd=False)

X_rec = tl.tucker_to_tensor((G, fac))

x, x_test, y_true, y_test = tts(X_rec, faces.target, test_size=0.01, random_state=42)

model_lda = LDA(n_components=3)
model_lda.fit(x, y_true)
proj = model_lda.transform(x)
test_proj = model_lda.transform(x_test)

hits = [0]
for k in range(1,11):
  model_knn = knn(k)
  model_knn.fit(proj, y_true)
  y_pred = model_knn.predict(test_proj)
  hits.append(accuracy_score(y_test, y_pred))

plt.step(np.linspace(0, 10, 11), hits, where='mid', color='red', linewidth=2)
plt.grid(True)
plt.show()
