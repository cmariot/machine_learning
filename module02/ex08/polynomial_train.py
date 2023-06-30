import shutil
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class MyLR:

    def __init__(self, thetas, alpha=0.0001, max_iter=500_000):
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
        X_prime = np.c_[np.ones(m), x]
        return X_prime @ self.thetas

    def loss_elem_(self, y, y_hat):
        for arr in [y, y_hat]:
            if not isinstance(arr, np.ndarray):
                return None
        m = y.shape[0]
        if m == 0:
            return None
        if y.shape != (m, 1) or y_hat.shape != (m, 1):
            return None
        pred_sub_y = y_hat - y
        J_elem = np.dot(pred_sub_y.T, pred_sub_y)
        return J_elem

    def loss_(self, y, y_hat):
        J_elem = self.loss_elem_(y, y_hat)
        if J_elem is None:
            return None
        return (J_elem / (2 * y.shape[0]))[0, 0]

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
        return (X_prime.T @ (X_prime @ self.thetas - y)) / m

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
            self.thetas -= self.alpha * gradient
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

    def add_polynomial_features(self, x, power):
        if not isinstance(x, np.ndarray) or not isinstance(power, int):
            return None
        elif x.size == 0 or power < 0:
            return None
        if power == 0:
            return np.ones((x.size, 1))
        res = np.ones((x.size, power))
        for i in range(power):
            res[:, i] = x.ravel() ** (i + 1)
        return res


if __name__ == "__main__":

    linear_regression = MyLR([80, -10], 10e-10, 100_000)

    # Get the dataset in the file are_blue_pills_magics.csv
    try:
        dataset = pd.read_csv("../ressources/are_blue_pills_magics.csv")

    except Exception:
        print("Couldn't find the dataset file")
        exit(1)

    print(dataset, "\n")

    x = dataset['Micrograms'].values
    y = dataset['Score'].values.reshape(-1, 1)

    polynomial_x = linear_regression.add_polynomial_features(x, 6)

    # Loss = 18.13
    hypothesis_theta1 = np.array([[89.04720427],
                                  [-8.99425854]]
                                 ).reshape(-1, 1)

    # Loss = 26.988042374726316
    hypothesis_theta2 = np.array([[69.77316037],
                                  [1.49660362],
                                  [-1.21861482]]).reshape(-1, 1)

    # Loss = 27.87
    hypothesis_theta3 = np.array([[89.0],
                                  [-8.4],
                                  [0.8],
                                  [-0.1]]).reshape(-1, 1)

    # Loss = 45
    hypothesis_theta4 = np.array([[-19.9],
                                  [160.4],
                                  [-78.6],
                                  [13.6],
                                  [-0.8]]
                                 ).reshape(-1, 1)

    # Loss = 12.83
    hypothesis_theta5 = np.array([[1140],
                                  [-1850],
                                  [1110],
                                  [-305.2],
                                  [39.3],
                                  [-1.9]]
                                 ).reshape(-1, 1)

    hypothesis_theta6 = np.array([[9110],
                                  [-18015],
                                  [13400],
                                  [-4935],
                                  [966],
                                  [-96.4],
                                  [3.86]]
                                 ).reshape(-1, 1)

    hypothesis_thetas = [hypothesis_theta1, hypothesis_theta2,
                         hypothesis_theta3, hypothesis_theta4,
                         hypothesis_theta5, hypothesis_theta6]

    thetas = []
    mse_scores = []

    # Trains six separate Linear Regression models with polynomial
    # hypothesis with degrees ranging from 1 to 6
    # Plots the 6 models and the data points on the same figure.
    # Use lineplot style for the models and scaterplot for the data points.
    # Add more prediction points to have smooth curves for the models.
    fig, ax = plt.subplots(2, 3)

    for i in range(1, 7):
        print("Training model {} / 6\n".format(i))

        linear_regression.thetas = hypothesis_thetas[i - 1]
        current_x = polynomial_x[:, :i]
        linear_regression.fit_(current_x, y)
        y_hat = linear_regression.predict_(current_x)

        thetas.append(linear_regression.thetas)
        mse_scores.append(linear_regression.mse_(y, y_hat))

        # Plots the data points
        ax[(i - 1) // 3][(i - 1) % 3].scatter(x, y, color='blue')

        # Plots the model curve
        min_x = np.min(x)
        max_x = np.max(x)
        continuous_x = np.linspace(min_x, max_x, 100)
        predicted_x = linear_regression.add_polynomial_features(continuous_x,
                                                                i)
        predicted_y = linear_regression.predict_(predicted_x)
        ax[(i - 1) // 3][(i - 1) % 3].plot(continuous_x, predicted_y,
                                           color='orange')
        # Add title and axis names
        ax[(i - 1) // 3][(i - 1) % 3].set_title(
            "Degree {}, score : {}".format(i, mse_scores[i - 1]))
        ax[(i - 1) // 3][(i - 1) % 3].set_xlabel("Micrograms")
        ax[(i - 1) // 3][(i - 1) % 3].set_ylabel("Score")

        # Compute Loss
        loss = linear_regression.loss_(y, y_hat)
        print()
        print("Loss {} : {}".format(i, loss))
        print("Thetas : {}".format(linear_regression.thetas))
        print()

    plt.show()

    for i in range(6):
        print("Model {} :".format(i + 1))
        print("Thetas : {}".format(thetas[i]))
        print("MSE : {}\n".format(mse_scores[i]))

    # Plots a bar plot showing the MSE score of the models in function of
    # the polynomial degree of the hypothesis,
    plt.bar([1, 2, 3, 4, 5, 6], mse_scores)
    plt.xlabel("Polynomial degree")
    plt.ylabel("MSE")
    plt.show()
