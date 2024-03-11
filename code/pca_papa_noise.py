import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

img = plt.imread('data/papa_noise.png')
plt.imshow(img, cmap='gray')
plt.show()

model = PCA(0.99)
model.fit(img)

print("Number of Components:", len(model.explained_variance_ratio_))
plt.plot(np.cumsum(model.explained_variance_ratio_))
plt.show()

img_rec = model.inverse_transform(model.fit_transform(img))
plt.imshow(img_rec, cmap='gray')
plt.show()