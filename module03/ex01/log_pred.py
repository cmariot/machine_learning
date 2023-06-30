import numpy as np


def sigmoid_(x):
    """
    Compute the sigmoid of a vector.
    """
    if not isinstance(x, np.ndarray):
        print("Error: x is not a np.array.")
        return None
    m = x.shape[0]
    if m == 0 or (x.shape != (m, ) and x.shape != (m, 1)):
        print("Error: x is not a np.array of shape (m, 1).")
        print(x.shape)
        return None
    return 1. / (1. + np.exp(-x))


def logistic_predict_(x, theta):

    for arr in [x, theta]:
        if not isinstance(arr, np.ndarray):
            print("Error: x or theta is not a np.array.")
            return None

    m = x.shape[0]
    n = x.shape[1]

    if theta.shape != (n + 1, 1):
        print("Error: theta is not a np.array of shape (n + 1, 1).")
        return None

    X_prime = np.c_[np.ones((m, 1)), x]
    y_hat = np.zeros((m, 1))
    dot = (np.dot(X_prime, theta)).reshape((m, 1))
    for i in range(m):
        y_hat[i] = sigmoid_(dot[i])
        if y_hat[i] is None:
            return None

    return y_hat


if __name__ == "__main__":

    x = np.array([4]).reshape((-1, 1))
    theta = np.array([[2], [0.5]])
    print(logistic_predict_(x, theta))

    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]])
    print(logistic_predict_(x2, theta2))

    x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    print(logistic_predict_(x3, theta3))
