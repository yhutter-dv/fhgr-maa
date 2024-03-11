import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn import datasets

wine = datasets.load_wine()
x = wine.data
y_true = wine.target

model_lda = LDA(n_components=2)
model_lda.fit(x, y_true)
x_proj_lda = model_lda.transform(x)
y_pred = model_lda.predict(x)

model_pca = PCA(n_components=2)
model_pca.fit(x, y_true)
x_proj_pca = model_pca.transform(x)

plt.scatter(x_proj_lda[:,0], x_proj_lda[:,1], c=y_pred, alpha=0.8)
plt.show()

plt.scatter(x_proj_pca[:,0], x_proj_pca[:,1], c=y_true, alpha=0.8)
plt.show()

