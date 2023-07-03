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


class TestLogisticPredict(unittest.TestCase):

    def test_logistic_predict_with_non_array_input(self):
        x = 5
        theta = np.array([[2], [0.5]])
        self.assertIsNone(logistic_predict_(x, theta))

    def test_logistic_predict_with_non_array_theta(self):
        x = np.array([4]).reshape((-1, 1))
        theta = 5
        self.assertIsNone(logistic_predict_(x, theta))

    def test_logistic_predict_with_empty_input(self):
        x = np.array([])
        theta = np.array([[2], [0.5]])
        self.assertIsNone(logistic_predict_(x, theta))

    def test_logistic_predict_with_empty_theta(self):
        x = np.array([4]).reshape((-1, 1))
        theta = np.array([])
        self.assertIsNone(logistic_predict_(x, theta))

    def test_logistic_predict_with_wrong_shape_input(self):
        x = np.array([[4, 5], [7.16, 8.23]])
        theta = np.array([[2], [0.5]])
        self.assertIsNone(logistic_predict_(x, theta))

    def test_logistic_predict_with_wrong_shape_theta(self):
        x = np.array([4]).reshape((-1, 1))
        theta = np.array([[2, 0.5], [1, 0.2]])
        self.assertIsNone(logistic_predict_(x, theta))

    def test_logistic_predict_with_one_sample(self):
        x = np.array([4]).reshape((-1, 1))
        theta = np.array([[2], [0.5]])
        self.assertTrue(np.allclose(logistic_predict_(x, theta),
                                    np.array([[0.98201379]])))

    def test_logistic_predict_with_multiple_samples(self):
        x = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
        theta = np.array([[2], [0.5]])
        self.assertTrue(np.allclose(logistic_predict_(x, theta),
                                    np.array([[0.98201379],
                                              [0.99624161],
                                              [0.97340301],
                                              [0.99875204],
                                              [0.90720705]])))

    def test_logistic_predict_with_multiple_features(self):
        x = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
        theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
        self.assertTrue(np.allclose(logistic_predict_(x, theta),
                                    np.array([[0.03916572],
                                              [0.00045262],
                                              [0.2890505]])))


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

    unittest.main()
