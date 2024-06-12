import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac

X = np.arange(24).reshape((3, 4, 2))
X = X + 1.0

for rank in range(1, 5):
    factors = parafac(X, rank)
    # print(factors.factors)
    # print(factors.weights)
    X_rec = tl.kruskal_to_tensor(factors)
    print("Rank: ", rank, "Error: ", tl.norm(X - X_rec))
