import numpy as np
import matplotlib.pyplot as plt

def plot_faces(data, x=3, y=6, rnd=True):
    if rnd == True:
        idx = np.absolute(1000*np.random.normal(0, 1, x*y)).astype(int)%400
    else:
        idx = np.arange(x*y)
    fig, axs = plt.subplots(x, y)
    for i, ax in enumerate(axs.flat):
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.imshow(data[idx[i]], cmap='gray')
    plt.show()
