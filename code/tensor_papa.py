import tensorly as tl
import matplotlib.pyplot as plt
from tensorly.decomposition import parafac

img = plt.imread('./data/papa.png')
rank = 200
factors = parafac(img, rank)
img_rec = tl.kruskal_to_tensor(factors)
plt.imshow(img_rec, cmap='gray')
plt.show()
