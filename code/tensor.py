import numpy as np
import tensorly as tl

u = np.array([3, 1, 2])
v = np.array([4, 0, 3])
w = np.array([2, 3, 4])

X = np.einsum('i,j,k->ijk', u, v, w)

A = np.array([[2, 1], [1, 3]])
B = np.array([[3, 3], [-2, 4]])
kronecker = tl.tenalg.kronecker((A, B))
kathri = tl.tenalg.khatri_rao((A, B))
hadamar = A * B
print(kronecker)
print(kathri)
print(hadamar)