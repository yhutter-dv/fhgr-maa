
if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.linear_model import LinearRegression as lr
    from sklearn.model_selection import train_test_split as tts
    from sklearn.metrics import root_mean_squared_error as rmse
    from sklearn.metrics import r2_score 
    import matplotlib.pyplot as plt

    # Load dataset
    data = datasets.fetch_california_housing()
    x = data.data[:,2].reshape((-1, 1))
    y = data.data[:,3] 

    # Split into test and train data
    test_size = 0.2
    x_train, x_test, y_train, y_test = tts(x, y, test_size=test_size, random_state=42)

    model = lr()
    model.fit(x_train, y_train)

    # Note that this is only done to get the regression curve. It is NOT used to get an actual predicted result.
    # For that we would need to pass in the actual test data.
    y_train_pred = model.predict(x_train)
    print("MSE (train):", rmse(y_train, y_train_pred)**2)
    print("R2 Score (train):", r2_score(y_train, y_train_pred))
    
    # Test with test data
    y_test_pred = model.predict(x_test)
    print("MSE (test):", rmse(y_test, y_test_pred)**2)
    print("R2 Score (test):", r2_score(y_test, y_test_pred))

    # Plot result
    plt.xlabel("Number of Rooms")
    plt.ylabel("Number of Bed Rooms")
    plt.grid(True)

    plt.plot(x_train, y_train, 'ko', label="train")
    plt.plot(x_test, y_test, 'ro', label="test")
    plt.plot(x_test, y_test_pred, 'b-')
    plt.legend(loc=2)
    plt.legend(loc=2)
    plt.show()

