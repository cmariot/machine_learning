import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sys import exit
from sklearn.metrics import mean_squared_error


class MyLinearRegression():

    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        if (not isinstance(thetas, np.ndarray) or thetas.shape != (2, 1)):
            return None
        self.thetas = thetas
        if not isinstance(alpha, float) or alpha < 0.0:
            return None
        self.alpha = alpha
        if not isinstance(max_iter, int) or max_iter < 0:
            return None
        self.max_iter = max_iter

    def fit_(self, x, y):
        for arr in [x, y]:
            if not isinstance(arr, np.ndarray):
                return None
            if arr.size == 0:
                return None
        m = x.shape[0]
        if x.shape != (m, 1) or y.shape != (m, 1):
            return None
        gradient = np.zeros((2, 1))
        x_size = x.size
        Xprime = np.c_[np.ones(x_size), x]
        XprimeT = Xprime.T
        for _ in range(self.max_iter):
            gradient = XprimeT @ (Xprime @ self.thetas - y) / x_size
            if gradient is None:
                return None
            elif all(val == [0.0] for val in gradient):
                break
            self.thetas = self.thetas - self.alpha * gradient
            print(" {:2.2f} %" .format(_ / self.max_iter * 100), end="\r")
        return self.thetas

    def predict_(self, x):
        if not isinstance(x, np.ndarray):
            return None
        m = x.shape[0]
        if m == 0 or x.shape != (m, 1):
            return None
        X = np.c_[np.ones(m), x]
        return X @ self.thetas

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
        J_elem = self.loss_elem_(y, y_hat)
        if J_elem is None:
            return None
        return np.mean(J_elem) / 2

    def mse_elem(self, y, y_hat) -> np.ndarray:
        return (y_hat - y) ** 2

    def mse_(self, y, y_hat) -> float:
        """
            Description:
                Calculate the MSE between the predicted output and the real
                output.
            Args:
                y: has to be a numpy.array, a vector of dimension m * 1.
                y_hat: has to be a numpy.array, a vector of dimension m * 1.
            Returns:
                mse: has to be a float.
                None if there is a matching dimension problem.
            Raises:
                This function should not raise any Exceptions.
        """
        if any(not isinstance(_, np.ndarray) for _ in [y, y_hat]):
            return None
        m = y.size
        if m == 0 or y.shape != (m, 1) or y_hat.shape != (m, 1):
            return None
        J_elem = self.mse_elem(y, y_hat)
        return J_elem.mean()


def get_dataset():
    # Read the dataset
    try:
        return pd.read_csv("are_blue_pills_magics.csv")
    except Exception:
        print("Error : Cannot read the dataset")
        exit(1)


def plot_linear_regression(Xpill, Yscore, y_hat):
    plt.title("Linear regression of space driving score\n"
              + "depending on quantity of blue pill")
    plt.plot(Xpill, Yscore, 'bo')
    plt.plot(Xpill, y_hat, 'xg--')
    for i in range(Xpill.shape[0]):
        plt.plot([Xpill[i], Xpill[i]], [Yscore[i], y_hat[i]], 'r--')
    plt.xlabel("Quantity of blue pill (in micrograms)")
    plt.ylabel("Space driving score")
    plt.legend(["Real values", "Predicted values"])
    plt.show()


def plot_loss_evolution(Xpill, Yscore):
    some_thetas_zero = [85, 87.5, 90, 92.5, 95]
    for theta_zero in some_thetas_zero:
        loss_values = []
        theta_one_values = np.arange(-14, -4, 0.01)
        for theta_one in theta_one_values:
            linear_model = MyLinearRegression(
                np.array([[theta_zero], [theta_one]]))
            y_hat = linear_model.predict_(Xpill)
            loss = linear_model.loss_(Yscore, y_hat)
            loss_values.append(loss)
        plt.plot(theta_one_values, loss_values)

    plt.xlabel("θ1 values")
    plt.ylabel("Loss function J")
    plt.legend(some_thetas_zero)
    plt.title("Evolution of the loss function J as a function of θ1\n"
              + "for different values of θ0")
    plt.ylim(10, 150)
    plt.grid()
    plt.show()


def compute_mse(Yscore, Xpill):
    linear_model1 = MyLinearRegression(np.array([[89.0], [-8]]))
    linear_model2 = MyLinearRegression(np.array([[89.0], [-6]]))
    Y_model1 = linear_model1.predict_(Xpill)
    Y_model2 = linear_model2.predict_(Xpill)
    print(Y_model1, Y_model2)
    print(MyLinearRegression.mse_(Yscore, Y_model1))
    # 57.60304285714282
    print(mean_squared_error(Yscore, Y_model1))
    # 57.603042857142825
    print(MyLinearRegression.mse_(Yscore, Y_model2))
    # 232.16344285714285
    print(mean_squared_error(Yscore, Y_model2))
    # 232.16344285714285

if __name__ == "__main__":

    data = get_dataset()

    # Variables init
    Xpill = np.array(data["Micrograms"]).reshape(-1, 1)
    Yscore = np.array(data["Score"]).reshape(-1, 1)
#    thetas = [np.array([[1.], [1.]])]
#    linear_model = MyLinearRegression(thetas[0], 0.001, 1_500_000)
#
#    # Train the model and evaluate the predicted values
#    linear_model.fit_(Xpill, Yscore)
#    y_hat = linear_model.predict_(Xpill)
#    loss = linear_model.loss_(Yscore, y_hat)
#
#    print("theta0 = {:f}, theta1 = {:f}, loss = {:f}".format(
#        linear_model.thetas[0][0], linear_model.thetas[1][0], loss))
#
#    # Plot the real values, the predicted values and the linear regression
#    plot_linear_regression(Xpill, Yscore, y_hat)
#
#    # Compute and plot evolution of the loss function J as a function of θ1
#    # for different values of θ0
#    plot_loss_evolution(Xpill, Yscore)

    compute_mse(Yscore, Xpill)
