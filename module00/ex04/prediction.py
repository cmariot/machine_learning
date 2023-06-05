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
    if x.shape != (m, ) or theta.shape != (2, ):
        return None

    # Compute y_hat, the vector of prediction with a matrix multiplication
    y_hat = X.dot(theta)

    return y_hat


if __name__ == "__main__":

    x = np.arange(1, 6)
    # x = [1, 2, 3, 4, 5]

    theta1 = np.array([5, 0])
    y_hat = predict_(x, theta1)
    print(y_hat)
    # array([5., 5., 5., 5., 5.])

    theta2 = np.array([0, 1])
    y_hat = predict_(x, theta2)
    print(y_hat)
    # array([1., 2., 3., 4., 5.])

    theta3 = np.array([5, 3])
    y_hat = predict_(x, theta3)
    print(y_hat)
    # array([ 8., 11., 14., 17., 20.])

    theta4 = np.array([-3, 1])
    y_hat = predict_(x, theta4)
    print(y_hat)
    # array([-2., -1.,  0.,  1.,  2.])
