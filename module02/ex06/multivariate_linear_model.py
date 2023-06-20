import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import shutil


class MyLR:

    def __init__(self, thetas, alpha=0.0001, max_iter=500000):
        if not isinstance(alpha, float) or alpha < 0.0:
            return None
        elif not isinstance(max_iter, int) or max_iter < 0:
            return None
        self.thetas = np.array(thetas)
        self.alpha = alpha
        self.max_iter = max_iter

    def predict_(self, x):
        if not isinstance(x, np.ndarray):
            return None
        elif x.size == 0:
            return None
        m = x.shape[0]
        n = x.shape[1]
        if self.thetas.shape != (n + 1, 1):
            return None
        X_prime = np.concatenate((np.ones((m, 1)), x), axis=1)
        y_hat = np.dot(X_prime, self.thetas)
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
        J_elem = np.square(y_hat - y)
        return J_elem

    def loss_(self, y, y_hat):
        J_elem = self.loss_elem_(y, y_hat)
        if J_elem is None:
            return None
        J_value = np.mean(J_elem) / 2
        return J_value

    def gradient_(self, x, y):
        for array in [x, y]:
            if not isinstance(array, np.ndarray):
                return None
        m, n = x.shape
        if m == 0 or n == 0:
            return None
        elif y.shape != (m, 1):
            return None
        elif self.thetas.shape != (n + 1, 1):
            return None
        X_prime = np.c_[np.ones(m), x]
        return (1 / m) * (X_prime.T.dot(X_prime.dot(self.thetas) - y))

    def ft_progress(self, iterable,
                    length=shutil.get_terminal_size().columns - 2,
                    fill='█',
                    empty='░',
                    print_end='\r'):
        total = len(iterable)
        start = time.time()
        for i, item in enumerate(iterable, start=1):
            elapsed_time = time.time() - start
            eta = elapsed_time * (total / i - 1)
            current_percent = (i / total) * 100
            filled_length = int(length * i / total)
            if eta == 0.0:
                eta_str = '[DONE]    '
            elif eta < 60:
                eta_str = f'[ETA {eta:.0f} s]'
            elif eta < 3600:
                eta_str = f'[ETA {eta / 60:.0f} m]'
            else:
                eta_str = f'[ETA {eta / 3600:.0f} h]'
            percent_str = f'[{current_percent:6.2f} %] '
            progress_str = str(fill * filled_length
                               + empty * (length - filled_length))
            counter_str = f' [{i:>{len(str(total))}}/{total}] '
            if elapsed_time < 60:
                et_str = f' [Elapsed-time {elapsed_time:.2f} s]'
            elif elapsed_time < 3600:
                et_str = f' [Elapsed-time {elapsed_time / 60:.2f} m]'
            else:
                et_str = f' [Elapsed-time {elapsed_time / 3600:.2f} h]'
            bar = ("\033[F\033[K " + progress_str + "\n"
                   + et_str
                   + counter_str
                   + percent_str
                   + eta_str)
            print(bar, end=print_end)
            yield item

    def fit_(self, x, y):
        for arr in [x, y]:
            if not isinstance(arr, np.ndarray):
                return None
        m, n = x.shape
        if m == 0 or n == 0:
            return None
        if y.shape != (m, 1):
            return None
        elif self.thetas.shape != ((n + 1), 1):
            return None
        for _ in self.ft_progress(range(self.max_iter)):
            gradient = self.gradient_(x, y)
            if gradient is None:
                return None
            if all(__ == 0. for __ in gradient):
                break
            self.thetas = self.thetas - self.alpha * gradient
        print()
        return self.thetas

    def mse_elem(self, y, y_hat) -> np.ndarray:
        return (y_hat - y) ** 2

    def mse_(self, y, y_hat) -> float:
        if any(not isinstance(_, np.ndarray) for _ in [y, y_hat]):
            return None
        m = y.shape[0]
        if m == 0 or y.shape != (m, 1) or y_hat.shape != (m, 1):
            return None
        J_elem = self.mse_elem(y, y_hat)
        return J_elem.mean()

    def minmax(x):
        if not isinstance(x, np.ndarray):
            return None
        if x.size == 0:
            return None
        if x.ndim != 1:
            x = x.reshape(-1)
        return (x - np.min(x)) / (np.max(x) - np.min(x))


if __name__ == "__main__":

    # Instanciate linear regression class
    linear_regression = MyLR([[0.0], [1.0]])

    # Open and read the file spacecraft_data.csv with pandas
    spacecraft_data = pd.read_csv("../ressources/spacecraft_data.csv")

    # Should I normalize the features ?

    # PART ONE : Univariate linear regression
    features = ["Age", "Thrust_power", "Terameters"]
    for feature in features:
        linear_regression.thetas = np.array([[500.0], [1.0]])
        x = np.array(spacecraft_data[feature]).reshape(-1, 1)
        y = np.array(spacecraft_data["Sell_price"]).reshape(-1, 1)
        linear_regression.fit_(x, y)
        y_hat = linear_regression.predict_(x)

        # print MSE
        print("MSE : {}".format(linear_regression.mse_(y, y_hat)))

        # Plot the data and the prediction
        plt.scatter(x, y)
        plt.scatter(x, y_hat, color="orange", marker=".")
        plt.title(
            "Sell price of a spacecraft depending on its {}".format(feature))
        plt.xlabel(feature)
        plt.ylabel("Sell price")
        plt.grid()
        plt.legend(["Data", "Prediction"])
        plt.show()

    # PART TWO : Multivariate linear regression
    features = ["Age", "Thrust_power", "Terameters"]
    x = np.array(spacecraft_data[features])
    y = np.array(spacecraft_data["Sell_price"]).reshape(-1, 1)

    # Print MSE with initial thetas (1.0)
    linear_regression.thetas = np.array([[1.0], [1.0], [1.0], [1.0]])
    y_hat = linear_regression.predict_(x)
    print("MSE : {}".format(linear_regression.mse_(y, y_hat)))

    # Update thetas with appropriate values
    linear_regression.thetas = np.array([[385.21139513],
                                         [-24.33149116],
                                         [5.67045772],
                                         [-2.66684314]])
    linear_regression.alpha = 1e-5
    linear_regression.max_iter = 900000

    # Train the model and predict y_hat
    linear_regression.fit_(x, y)
    y_hat = linear_regression.predict_(x)
    print("thetas : {}".format(linear_regression.thetas))

    # Plot the data and the prediction
    for feature in features:
        plt.scatter(x[:, features.index(feature)], y)
        plt.scatter(x[:, features.index(feature)], y_hat,
                    color="orange",
                    marker=".")
        plt.title(
            "Sell price of a spacecraft depending on its {}".format(feature))
        plt.xlabel(feature)
        plt.ylabel("Sell price")
        plt.grid()
        plt.legend(["Data", "Prediction"])
        plt.show()

    # PART THREE : Mean squared error
    print("MSE : {}".format(linear_regression.mse_(y, y_hat)))
