import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union


class MyLinearRegression():
    """
        Description:
                My personnal linear regression class to fit like a boss.
    """

    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        # Args checking :
        if not isinstance(thetas, np.ndarray):
            return None
        if thetas.shape != (2, 1) and thetas.shape != (2,):
            return None
        if not isinstance(alpha, float):
            return None
        if not isinstance(max_iter, int):
            return None
        if alpha < 0:
            return None
        if max_iter < 0:
            return None

        self.thetas = thetas
        self.alpha = alpha
        self.max_iter = max_iter

    def fit_(self, x, y):
        for arr in [x, y, self.thetas]:
            if not isinstance(arr, np.ndarray):
                return None
            if arr.size == 0:
                return None

        # Check if x and y have compatible shapes
        m = x.shape[0]
        if x.shape != y.shape or x.shape[1] != 1 \
                or self.thetas.shape != (2, 1):
            return None

        # Reshape the x array to be a matrix of shape m * 2
        X = np.c_[np.ones((m, 1)), x]
        X = X.reshape((m, 2))

        # Gradient descent
        self.thetas = self.thetas.reshape((2, 1))
        for _ in range(self.max_iter):
            gradient = np.zeros((2, 1))
            gradient[0] = np.sum((X.dot(self.thetas) - y)) / m
            gradient[1] = np.sum((X.dot(self.thetas) - y) * x) / m
            self.thetas = self.thetas - self.alpha * gradient

        return self.thetas

    def predict_(self, x) -> Union[np.ndarray, None]:
        """
        Computes the vector of prediction y_hat from two non-empty numpy.array.
        """
        if not isinstance(x, np.ndarray):
            return None
        elif x.size == 0 or x.shape[1] != 1:
            return None
        X = np.c_[np.ones(x.shape[0]), x]
        y_hat = X.dot(self.thetas)

        return y_hat

    def loss_elem_(self, y, y_hat):
        for arr in [y, y_hat]:
            if not isinstance(arr, np.ndarray):
                return None
        m = y.shape[0]
        if m == 0:
            return None
        if y.shape != (m, 1) or y_hat.shape != (m, 1):
            return None
        return (y_hat - y) ** 2

    def loss_(self, y, y_hat):
        # Compute the loss by calling loss_elem_
        J_elem = self.loss_elem_(y, y_hat)
        if J_elem is None:
            return None
        # Compute the mean of J_elem
        J_value = np.mean(J_elem)
        return J_value / 2


if __name__ == "__main__":

    # Read the dataset
    data = pd.read_csv("are_blue_pills_magics.csv")
    Xpill = np.array(data["Micrograms"]).reshape(-1, 1)
    Yscore = np.array(data["Score"]).reshape(-1, 1)

    thetas = [np.array([[1.], [1.]])]

    # Perform a linear regression
    linear_model = MyLinearRegression(thetas[0], 0.001, 150)

    for _ in range(1000):
        linear_model.fit_(Xpill, Yscore)
        thetas.append(linear_model.thetas)

    y_hat = linear_model.predict_(Xpill)
    loss = linear_model.loss_(Yscore, y_hat)

    print("theta0 = {:f}, theta1 = {:f}, loss = {:f}".format(
        linear_model.thetas[0][0], linear_model.thetas[1][0], loss))

    # Plot the result
    plt.plot(Xpill, Yscore, 'bo')
    plt.plot(Xpill, y_hat, 'xg--')
    plt.xlabel("Quantity of blue pill (in micrograms)")
    plt.ylabel("Space driving score")
    plt.legend(["Strue", "Spredicted"])
    plt.title("Linear regression of space driving score\n"
              + "depending on quantity of blue pill")

    plt.show()

    # Plot evolution of the loss function J as a function of θ1
    # for different values of θ0
    some_thetas_zero = [85, 87.5, 90, 92.5, 95]
    for theta_zero in some_thetas_zero:
        loss_values = []
        theta_one_values = np.arange(-14, -4, 0.01)
        for theta_one in theta_one_values:
            linear_model = MyLinearRegression(
                np.array([[theta_zero], [theta_one]]))
            y_hat = linear_model.predict_(Xpill)
            loss_values.append(linear_model.loss_(Yscore, y_hat))

        plt.plot(theta_one_values, loss_values)
    plt.xlabel("θ1 values")
    plt.ylabel("Loss function J")
    plt.legend(some_thetas_zero)
    plt.title("Evolution of the loss function J as a function of θ1\n"
              + "for different values of θ0")
    plt.ylim(10, 150)
    plt.grid()
    plt.show()
