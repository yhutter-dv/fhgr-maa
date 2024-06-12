from mytensor import gen_tensor_three_feature as gen3f
from mytensor import plot_uvw_three_feature as plot3f
from tensorly.decomposition import parafac

X = gen3f()
w_, fac = parafac(X, 3)
plot3f(fac)
