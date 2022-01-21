import numpy as np
from basis_function import RBF, Polynomial
from linear_regression import BayesianLinearRegression
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
np.random.seed(100)


def load_data(path):
    """

    :param path: you need to use your path
    :return: data, label
    """
    f = open(path)
    dataset = []
    for line in f.readlines():
        line = line.strip().split()
        line = list(map(float, line))
        dataset.append(line)
    dataset = np.array(dataset)
    np.random.shuffle(dataset)
    return dataset[:, 0], dataset[:, 1]


def main(path, alpha, beta):
    data, label = load_data(path)
    x_train = data[:150]
    y_train = label[:150]
    x_test = data[150:]
    y_test = label[150:]
    rbf = RBF(np.linspace(0, 1, 50), 0.1)
    X_train = rbf(x_train)
    X_test = rbf(x_test)
    model = BayesianLinearRegression(alpha, beta)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    x = np.linspace(0, 1, 100)
    X = rbf(x)
    # kernel = DotProduct() + WhiteKernel()
    # model = GaussianProcessRegressor(kernel=kernel, alpha=1e-3)
    # model.fit(x_train.reshape(-1, 1), y_train)
    plt.scatter(x_train, y_train, c='blue', label='train_data')
    plt.scatter(x_test, y_test, c='orange', label='test_data')
    plt.plot(x, model.predict(X))
    plt.show()
    test_error = np.square(y_pred - y_test).sum() / y_pred.shape[-1]
    return model, test_error


if __name__ == '__main__':
    path = "../data/ex0.txt"
    model, test_error = main(path, 1e-4, 50)
    print(f"test_error={test_error}")
