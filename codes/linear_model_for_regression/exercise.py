import numpy as np
from basis_function import RBF, Polynomial
from linear_regression import BayesianLinearRegression
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel



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
    train = dataset[: 150]
    np.random.shuffle(train)
    test = dataset[150:]
    return train[:, 0], train[:, 1], test[:, 0], test[:, 1]


def main(path, alpha, beta):
    x_train, y_train, x_test, y_test = load_data(path)
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
