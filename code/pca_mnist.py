from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

from myplot import plot_digits
import matplotlib.pyplot as plt

digits = load_digits()
digits.data.shape

plot_digits(digits.data)
plt.show()

model = PCA(n_components=2)
d_proj = model.fit_transform(digits.data)
print(model.explained_variance_ratio_)

d_recov = model.inverse_transform(d_proj)
plot_digits(d_recov)
plt.show()