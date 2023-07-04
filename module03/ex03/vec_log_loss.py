import numpy as np
import unittest
import sklearn.metrics as skm


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


def vec_log_loss_(y, y_hat, eps=1e-15):
    """
        Compute the logistic loss value.
        Args:
            y: has to be an numpy.ndarray, a vector of shape m * 1.
            y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
            eps: epsilon (default=1e-15)
        Returns:
            The logistic loss value as a float.
            None on any error.
        Raises:
            This function should not raise any Exception.
    """

    if not all(isinstance(x, np.ndarray) for x in [y, y_hat]):
        return None

    m = y.shape[0]
    n = y.shape[1]

    if (m == 0 or n == 0):
        return None
    elif y_hat.shape != (m, n):
        return None

    try:
        # Change the values of y_hat to avoid math error
        y_hat[(y_hat == 0)] = eps
        y_hat[(y_hat == 1)] = 1 - eps
        dot1 = y.T.dot(np.log(y_hat))
        dot2 = (1 - y).T.dot(np.log(1 - y_hat))
        return ((dot1 + dot2) / -m)[0][0]

    except Exception:
        return None


class TestLogLoss(unittest.TestCase):

    def test_example_one(self):
        y1 = np.array([1]).reshape((-1, 1))
        x1 = np.array([4]).reshape((-1, 1))
        theta1 = np.array([[2], [0.5]])
        y_hat1 = logistic_predict_(x1, theta1)
        self.assertEqual(vec_log_loss_(y1, y_hat1),
                         skm.log_loss(y1, y_hat1, eps=1e-15, labels=[0, 1]))

    def test_example_two(self):
        y2 = np.array([[1], [0], [1], [0], [1]])
        x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
        theta2 = np.array([[2], [0.5]])
        y_hat2 = logistic_predict_(x2, theta2)
        self.assertEqual(vec_log_loss_(y2, y_hat2),
                         skm.log_loss(y2, y_hat2, eps=1e-15, labels=[0, 1]))

    def test_example_three(self):
        y3 = np.array([[0], [1], [1]])
        x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
        theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
        y_hat3 = logistic_predict_(x3, theta3)
        self.assertEqual(vec_log_loss_(y3, y_hat3),
                         skm.log_loss(y3, y_hat3, eps=1e-15, labels=[0, 1]))


if __name__ == "__main__":

    y1 = np.array([1]).reshape((-1, 1))
    x1 = np.array([4]).reshape((-1, 1))
    theta1 = np.array([[2], [0.5]])
    y_hat1 = logistic_predict_(x1, theta1)
    print(vec_log_loss_(y1, y_hat1))
    print(skm.log_loss(y1, y_hat1, eps=1e-15, labels=[0, 1]))

    y2 = np.array([[1], [0], [1], [0], [1]])
    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]])
    y_hat2 = logistic_predict_(x2, theta2)
    print(vec_log_loss_(y2, y_hat2))
    print(skm.log_loss(y2, y_hat2, eps=1e-15, labels=[0, 1]))

    y3 = np.array([[0], [1], [1]])
    x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    y_hat3 = logistic_predict_(x3, theta3)
    print(vec_log_loss_(y3, y_hat3))
    print(skm.log_loss(y3, y_hat3, eps=1e-15, labels=[0, 1]))
    
    unittest.main()
