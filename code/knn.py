from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score as score
import numpy as np

# Brutforce to get best k value
for k in range(1, 16):
    data = np.loadtxt('./data/credit.dat', delimiter=',')
    x_train, x_test, y_train, y_test = tts(data[:,:2], data[:,2],test_size=0.3)
    model = knn(n_neighbors=k)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(k, ":", score(y_test, y_pred))