import numpy as np


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


def log_gradient(x, y, theta):
    """
        Computes a gradient vector from three non-empty numpy.ndarray,
        with a for-loop.
        The three arrays must have compatible shapes.
        Args:
            x: has to be an numpy.ndarray, a matrix of shape m * n.
            y: has to be an numpy.ndarray, a vector of shape m * 1.
            theta: has to be an numpy.ndarray, a vector of shape (n + 1) * 1.
        Returns:
            The gradient as a numpy.ndarray, a vector of shape n * 1,
                containing the result of the formula for all j.
            None if x, y, or theta are empty numpy.ndarray.
            None if x, y and theta do not have compatible dimensions.
        Raises:
            This function should not raise any Exception.
    """

    try:
        if not all(isinstance(arr, np.ndarray) for arr in [x, y, theta]):
            return None

        m, n = x.shape

        if m == 0 or n == 0:
            return None
        elif y.shape != (m, 1) or theta.shape != ((n + 1), 1):
            return None

        y_hat = logistic_predict_(x, theta)
        if y_hat is None:
            return None

        gradient = np.zeros((n + 1, 1))
        gradient[0] = np.sum(y_hat - y)
        for j in range(1, n + 1):
            for i in range(1, m + 1):
                gradient[j] += (y_hat[i - 1] - y[i - 1]) * x[i - 1, j - 1]
        return gradient / m

    except Exception:
        return None


if __name__ == "__main__":

    y1 = np.array([1]).reshape((-1, 1))
    x1 = np.array([4]).reshape((-1, 1))
    theta1 = np.array([[2], [0.5]])
    gradient = log_gradient(x1, y1, theta1)
    print(gradient)
    # Output:
    # array([[-0.01798621],
    #        [-0.07194484]])

    # Example 2:
    y2 = np.array([[1], [0], [1], [0], [1]])
    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]])
    gradient = log_gradient(x2, y2, theta2)
    print(gradient)
    # Output:
    # array([[0.3715235 ],
    #        [3.25647547]])

    # Example 3:
    y3 = np.array([[0], [1], [1]])
    x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    gradient = log_gradient(x3, y3, theta3)
    print(gradient)
    # Output:
    # array([[-0.55711039],
    #        [-0.90334809],
    #        [-2.01756886],
    #        [-2.10071291],
    #        [-3.27257351]])
