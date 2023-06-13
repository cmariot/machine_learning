import numpy as np
from typing import Union


def add_intercept(x) -> Union[np.ndarray, None]:
    """Adds a column of 1's to the non-empty numpy.array x. Args:
          x: has to be a numpy.array of dimension m * n.
        Returns:
          X, a numpy.array of dimension m * (n + 1).
          None if x is not a numpy.array.
          None if x is an empty numpy.array.
        Raises:
          This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray):
        return None
    if x.size == 0:
        return None
    return np.c_[np.ones(x.shape[0]), x]


def predict_(
        x: np.ndarray, theta: np.ndarray) -> Union[np.ndarray, None]:

    """
        Computes the vector of prediction y_hat from two non-empty numpy.array.
        Args:
            x: has to be an numpy.array, a vector of dimension m * 1.
            theta: has to be an numpy.array, a vector of dimension 2 * 1.
        Returns:
            y_hat as a numpy.array, a vector of dimension m * 1.
            None if x and/or theta are not numpy.array.
            None if x or theta are empty numpy.array.
            None if x or theta dimensions are not appropriate.
        Raises:
            This function should not raise any Exceptions.
    """

    # Check if theta is a numpy.ndarray
    if not isinstance(theta, np.ndarray):
        return None

    # Add a column of ones to the vector x
    X = add_intercept(x)

    # If x is not a numpy.ndarray or x is empty, return None
    if X is None:
        return None

    # Check the dimension of x and theta
    m = x.shape[0]
    if x.shape != (m, 1) or theta.shape != (2, 1):
        return None

    # Compute y_hat, the vector of prediction with a matrix multiplication
    y_hat = X.dot(theta)

    return y_hat


def fit_(x, y, theta, alpha, max_iter):
    """
         Description:
            Fits the model to the training dataset contained in x and y.
        Args:
            x: has to be a numpy.ndarray, a vector of dimension m * 1:
                (number of training examples, 1).
            y: has to be a numpy.ndarray, a vector of dimension m * 1:
                (number of training examples, 1).
            theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
            alpha: has to be a float, the learning rate
            max_iter: has to be an int, the number of iterations done
                during the gradient descent
         Returns:
            new_theta: numpy.ndarray, a vector of dimension 2 * 1.
            None if there is a matching dimension problem.
         Raises:
            This function should not raise any Exception.
    """

    # Check if x, y, and theta are non empty numpy arrays
    for arr in [x, y, theta]:
        if not isinstance(arr, np.ndarray):
            return None
        if arr.size == 0:
            return None

    # Check if x and y have compatible shapes
    m = x.shape[0]
    if x.shape != y.shape or x.shape[1] != 1 or theta.shape != (2, 1):
        return None

    # Check if alpha and max_iter types and if they are positive
    if not isinstance(alpha, float) or not isinstance(max_iter, int):
        return None
    if alpha <= 0 or max_iter <= 0:
        return None

    # Reshape the x array to be a matrix of shape m * 2
    X = np.c_[np.ones((m, 1)), x]
    X = X.reshape((m, 2))

    # Gradient descent
    theta = theta.reshape((2, 1))
    gradient = np.zeros((2, 1))
    for _ in range(max_iter):
        gradient[0] = np.sum((X.dot(theta) - y)) / m
        gradient[1] = np.sum((X.dot(theta) - y) * x) / m
        theta = theta - alpha * gradient

    return theta


if __name__ == "__main__":
    x = np.array(
        [[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
    y = np.array(
        [[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
    theta = np.array([1, 1]).reshape((-1, 1))

    # Example 0:
    theta1 = fit_(x, y, theta, alpha=5e-8, max_iter=1500000)
    print(theta1)
    # Output:
    # array([[1.40709365],
    #        [1.1150909 ]])

    # Example 1:
    predict = predict_(x, theta1)
    print(predict)
    # Output:
    # array([[15.3408728 ],
    #        [25.38243697],
    #        [36.59126492],
    #        [55.95130097],
    #        [65.53471499]])
