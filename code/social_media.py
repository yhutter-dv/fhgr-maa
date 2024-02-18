import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as lr
from sklearn.metrics import mean_squared_error as mse


if __name__ == "__main__":
	data = np.loadtxt("./data/smp_data.dat", delimiter=",")
	
	# All elements in the first column
	# -1 = Unknown length and 1 column
	number_of_friends = data[:, 0].reshape(-1, 1)
	
	# All elements in the second column
	time_on_platform_seconds = data[:, 1]

	model = lr()

	model.fit(number_of_friends, time_on_platform_seconds)
	time_on_platform_seconds_pred = model.predict(number_of_friends)

	mse_result = mse(time_on_platform_seconds, time_on_platform_seconds_pred, squared=True)

	print(mse_result)

	plt.xlabel("Number of Friends")
	plt.ylabel("Time in Seconds spent on Social Media Plattform")
	plt.grid(True)
	plt.plot(number_of_friends, time_on_platform_seconds, 'ro')
	plt.plot(number_of_friends, time_on_platform_seconds_pred, 'b')
	#plt.show()

