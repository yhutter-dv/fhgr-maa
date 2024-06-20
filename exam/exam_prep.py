from sklearn import datasets
from sklearn.linear_model import LinearRegression as lr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker


def regression_boston_housing():
    data = datasets.fetch_california_housing()
    data_x = data.data[:, 2]
    data_y = data.data[:, 3]
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2)
    model = lr()
    model.fit(x_train.reshape((-1, 1)), y_train)
    y_pred = model.predict(x_test.reshape((-1, 1)))
    rmse(y_test, y_pred)
    r2_score(y_test, y_pred)

    plt.plot(x_test, y_test, "ro")
    plt.plot(x_test, y_pred, "b-")
    plt.show()


def wine_scatter():
    data = np.loadtxt("./data/wine.dat", delimiter=",")
    plt.scatter(data[:, 0], data[:, 1], c=data[:, 2])
    plt.show()


def wine_ellbow():
    data = np.loadtxt("./data/wine.dat", delimiter=",")
    x_data = data[:, :2]  # Spalten 0 und 1 = Koordinaten
    y_data = data[:, 2]  # Spalte 2 = korrekte Klassifikation
    inert = []
    for k in range(1, 11):
        model = KMeans(k)
        model.fit(x_data)
        inert.append(model.inertia_)
    x = np.linspace(1, 10, 10)
    plt.plot(x, inert, "b-")
    plt.plot(x, inert, "bo")
    plt.show()


def wine_kmeans():
    data = np.loadtxt("./data/wine.dat", delimiter=",")
    x_data = data[:, :2]  # Spalten 0 und 1 = Koordinaten
    model = KMeans(3)
    y_pred = model.fit_predict(x_data)
    plt.scatter(data[:, 0], data[:, 1], c=y_pred)
    plt.show()


def varriance_cumsum():
    varianz = [0.0, 80.0, 18.0, 1.0, 0.25, 0.25, 0.25, 0.25, 0.0]
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    plt.xlabel("Hauptkomponenten")
    plt.ylabel("Varianz [%]")
    plt.plot(x, np.cumsum(varianz), "r-")
    plt.show()


def iris_pca():
    iris = datasets.load_iris()
    x = iris.data
    y_true = iris.target
    model_pca = PCA(2)
    data_proj = model_pca.fit_transform(x)
    print("Varriance Ratio per Axis is: ", model_pca.explained_variance_ratio_)
    plt.scatter(data_proj[:, 0], data_proj[:, 1], c=y_true)
    plt.show()


def iris_pca_kmeans():
    iris = datasets.load_iris()
    x = iris.data
    y_true = iris.target
    model_pca = PCA(2)
    data_proj = model_pca.fit_transform(x)
    model_km = KMeans(3)
    y_pred = model_km.fit_predict(data_proj)
    plt.scatter(data_proj[:, 0], data_proj[:, 1], c=y_pred)
    plt.show()


def vector_scalar_product():
    u = [3, 4, 0, 1]
    v = [1, 0, -1, 2]
    w = [4, -3, 4, 0]
    print("Scalar Product u, v is: ", np.dot(u, v))
    print("Scalar Product u, w is: ", np.dot(u, w))
    print("Scalar Product v, w is: ", np.dot(v, w))


def transpose_matrix():
    A = [[3, 0, 4, 1], [2, 1, 3, 2]]
    print("Transposed Matrix A is: ", np.transpose(A))


def matrix_vector_mul():
    A = [[3, 0, 4, 1], [2, 1, 3, 2]]
    u = [2, 0, 4, 1]
    v = [1, 2]
    print(np.dot(A, u))
    print(np.dot(np.transpose(A), v))


def diadic_product():
    u = [2, 4, 1]
    v = [3, 1, 3, 2]
    print(np.outer(u, v))  # First possiblity
    print(np.einsum("i,j->ij", u, v))  # Second possibility


def diadic_product_tensor():
    u = [2, 4, 1]
    v = [3, 1, 3, 2]
    w = [5, 2]
    X = np.einsum("i,j,k->ijk", w, u, v)
    print(X)


def cancer_lda():
    cancer = datasets.load_breast_cancer()
    x, x_test, y_true, y_test = train_test_split(
        cancer.data, cancer.target, test_size=0.1, random_state=42
    )
    model = LDA(n_components=1)
    model.fit(x, y_true)
    x_proj = model.transform(x)
    x_test_proj = model.transform(x_test)
    hits = [0.0]
    for k in range(1, 11):
        model = knn(k)
        model.fit(x_proj, y_true)
        y_pred = model.predict(x_test_proj)
        # Wrong classifications
        for i in range(len(y_test)):
            if y_test[i] != y_pred[i]:
                print(
                    "Wrong classification for k ",
                    k,
                    cancer.target_names[y_test[i]],
                    "=>",
                    cancer.target_names[y_pred[i]],
                )
        hits.append(accuracy_score(y_test, y_pred))

    x_grid = np.linspace(0, 10, 11)
    plt.step(x_grid, hits, where="mid", color="red", linewidth=2)
    plt.show()


def kronecker_kathri_rao_hadamar_product():
    A = np.arange(6).reshape((2, 3)) + 1
    B = np.arange(6).reshape((2, 3)) + 7
    print("Kronechker: ", tl.tenalg.kronecker((A, B)))
    print("Kathri Rao: ", tl.tenalg.khatri_rao((A, B)))
    print("Hadamar: ", A * B)


def tensor_unfold():
    # Tensor with numbers from 1, 50 and dimension 5x2x5
    X = np.arange(50).reshape((5, 2, 5))
    X = X + 1
    print("Mode 0: ", tl.unfold(X, 0))
    print("Mode 1: ", tl.unfold(X, 1))
    print("Mode 2: ", tl.unfold(X, 2))


def tensor_tucker():
    # Tensor with random numbers and dimension 30x30x30
    X = np.random.rand(30, 30, 30)
    err = []
    for r in range(1, 51):
        G, fac = tucker(X, (r, r, r))
        X_rec = tl.tucker_to_tensor((G, fac))
        err.append(tl.norm(X - X_rec))
    plt.plot(err, "r-")
    plt.show()


def main():
    regression_boston_housing()
    # wine_scatter()
    # wine_ellbow()
    # wine_kmeans()
    # varriance_cumsum()
    # iris_pca()
    # iris_pca_kmeans()
    # vector_scalar_product()
    # transpose_matrix()
    # matrix_vector_mul()
    # diadic_product()
    # diadic_product_tensor()
    # cancer_lda()
    # kronecker_kathri_rao_hadamar_product()
    # tensor_unfold()
    # tensor_tucker()


if __name__ == "__main__":
    main()
