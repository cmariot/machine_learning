import numpy as np
import unittest


def sigmoid_(x):
    """
    Compute the sigmoid of a vector.
    Args:
        x: has to be a numpy.ndarray.
    Returns:
        The sigmoid value as a numpy.ndarray.
        None if an exception occurs.
    Raises:
        This function should not raise any Exception.
    """

    try:
        return 1 / (1 + np.exp(-x))

    except Exception:
        return None


def logistic_predict_(x, theta):
    """
    Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
    Args:
      x: has to be an numpy.ndarray, a vector of dimension m * n.
      theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
    Returns:
      y_hat as a numpy.ndarray, a vector of dimension m * 1.
      None if x or theta are empty numpy.ndarray.
      None if x or theta dimensions are not appropriate.
    Raises:
      This function should not raise any Exception.
    """

    if not all([isinstance(arr, np.ndarray) for arr in [x, theta]]):
        return None

    try:
        m, n = x.shape

        if m == 0 or n == 0:
            return None
        elif theta.shape != (n + 1, 1):
            return None

        X_prime = np.hstack((np.ones((m, 1)), x))
        y_hat = sigmoid_(X_prime.dot(theta))
        return y_hat

    except Exception:
        return None


def log_loss_(y, y_hat, eps=1e-15):
    """
    Computes the logistic loss value.
    Args:
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        eps: has to be a float, epsilon(default=1e-15)
    Returns:
        The logistic loss value as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """

    if not isinstance(y, np.ndarray):
        return None
    elif not isinstance(y_hat, np.ndarray):
        return None
    elif not isinstance(eps, float):
        return None

    m = y_hat.shape[0]
    n = y_hat.shape[1]

    if (m == 0 or n == 0):
        return None
    elif y.shape != (m, n):
        return None

    try:
        loss_elem = 0.0
        for i in range(m):
            ep = eps if y_hat[i][0] == 0 or y_hat[i][0] == 1 else 0
            loss_elem += y[i] * np.log(y_hat[i] + ep) + \
                (1 - y[i]) * np.log(1 - y_hat[i] + ep)
        loss = -1 / m * loss_elem
        return float(loss[0])

    except Exception:
        return None


class TestLogLoss(unittest.TestCase):

    def test_example_one(self):
        y1 = np.array([1]).reshape((-1, 1))
        x1 = np.array([4]).reshape((-1, 1))
        theta1 = np.array([[2], [0.5]])
        y_hat1 = logistic_predict_(x1, theta1)
        self.assertEqual(log_loss_(y1, y_hat1), 0.01814992791780973)

    def test_example_two(self):
        y2 = np.array([[1], [0], [1], [0], [1]])
        x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
        theta2 = np.array([[2], [0.5]])
        y_hat2 = logistic_predict_(x2, theta2)
        self.assertEqual(log_loss_(y2, y_hat2), 2.4825011602474483)

    def test_example_three(self):
        y3 = np.array([[0], [1], [1]])
        x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
        theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
        y_hat3 = logistic_predict_(x3, theta3)
        self.assertEqual(log_loss_(y3, y_hat3), 2.9938533108607057)
        # self.assertEqual(log_loss_(y3, y_hat3), 2.9938533108607053)


if __name__ == "__main__":

    y1 = np.array([1]).reshape((-1, 1))
    x1 = np.array([4]).reshape((-1, 1))
    theta1 = np.array([[2], [0.5]])
    y_hat1 = logistic_predict_(x1, theta1)
    print(log_loss_(y1, y_hat1))

    y2 = np.array([[1], [0], [1], [0], [1]])
    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]])
    y_hat2 = logistic_predict_(x2, theta2)
    print(log_loss_(y2, y_hat2))

    y3 = np.array([[0], [1], [1]])
    x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    y_hat3 = logistic_predict_(x3, theta3)
    print(log_loss_(y3, y_hat3))

    unittest.main()
