import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as lr
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

def test_data_plot(number_of_friends, time_on_platform_seconds):
	model = lr()

	# Learn from your data
	model.fit(number_of_friends, time_on_platform_seconds)
	
	# Based on your data predict a value
	time_on_platform_seconds_pred = model.predict(number_of_friends)

	mse_result = mse(time_on_platform_seconds, time_on_platform_seconds_pred, squared=True)
	print(mse_result)
	r2 = r2_score(time_on_platform_seconds, time_on_platform_seconds_pred)
	print(r2)
	plt.xlabel("Number of Friends")
	plt.ylabel("Time in Seconds spent on Social Media Plattform")
	plt.grid(True)
	plt.title("Testdata")
	plt.plot(number_of_friends, time_on_platform_seconds, 'ro')
	plt.plot(number_of_friends, time_on_platform_seconds_pred, 'b-')
	plt.show()
	
def train_data_plot(number_of_friends, time_on_platform_seconds):
	model = lr()

	# Learn from your data
	model.fit(number_of_friends, time_on_platform_seconds)
	
	# Based on your data predict a value
	time_on_platform_seconds_pred = model.predict(number_of_friends)

	mse_result = mse(time_on_platform_seconds, time_on_platform_seconds_pred, squared=True)
	print(mse_result)
	r2 = r2_score(time_on_platform_seconds, time_on_platform_seconds_pred)
	print(r2)
	plt.xlabel("Number of Friends")
	plt.ylabel("Time in Seconds spent on Social Media Plattform")
	plt.grid(True)
	plt.title("Traindata")
	plt.plot(number_of_friends, time_on_platform_seconds, 'ro')
	plt.plot(number_of_friends, time_on_platform_seconds_pred, 'b-')
	plt.show()

if __name__ == "__main__":
	data = np.loadtxt("./data/smp_data.dat", delimiter=",")
	
	# We need to reshape the data so if plays nicely with the LinearRegression from sklearn.
	# All elements in the first column -1 = Unknown length and 1 column
	number_of_friends = data[:, 0].reshape(-1, 1)
	
	# All elements in the second column
	time_on_platform_seconds = data[:, 1]

	x_train, x_test, y_train, y_test = train_test_split(number_of_friends, time_on_platform_seconds, test_size=0.2, random_state=42)
	test_data_plot(x_test, y_test)
	train_data_plot(x_train, y_train)
