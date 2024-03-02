import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

image_width = 512
image_height = 512
number_of_color_channels = 3
number_of_colors = 4
image_path = './data/papa_color.png'

img = plt.imread(image_path)
data = np.reshape(img, (image_width*image_height, number_of_color_channels))

model = KMeans(number_of_colors)
model.fit(data)
data_reduced = model.cluster_centers_[model.predict(data)]
reduced_image = np.reshape(data_reduced, (image_width, image_height, number_of_color_channels))
plt.imshow(reduced_image)
plt.show()
