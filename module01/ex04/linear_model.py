import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class MyLinearRegression():
    """
        Description:
            My personnal linear regression class to fit like a boss.
    """

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
        Xprime = np.c_[np.ones((m, 1)), x]
        XprimeT = Xprime.T
        gradient = np.zeros((2, 1))
        for _ in range(self.max_iter):
            gradient = np.matmul((XprimeT), (Xprime.dot(self.thetas) - y)) / m
            if gradient[0] == 0. and gradient[1] == 0.:
                break
            self.thetas = self.thetas - self.alpha * gradient
            print("{:2.2f} %".format(_ / self.max_iter * 100), end="\r")
        return self.thetas

    def predict_(self, x):
        """
        Computes the vector of prediction y_hat from two non-empty numpy.array.
        """
        if not isinstance(x, np.ndarray):
            return None
        m = x.shape[0]
        if m == 0 or x.shape != (m, 1):
            return None
        X = np.c_[np.ones(m), x]
        return X.dot(self.thetas)

    def loss_elem_(self, y, y_hat):
        for arr in [y, y_hat]:
            if not isinstance(arr, np.ndarray):
                return None
        m = y.shape[0]
        if m == 0:
            return None
        if y.shape != (m, 1) or y_hat.shape != (m, 1):
            return None
        return np.square(y_hat - y)

    def loss_(self, y, y_hat):
        J_elem = self.loss_elem_(y, y_hat)
        if J_elem is None:
            return None
        J_value = np.mean(J_elem) / 2
        return J_value


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
