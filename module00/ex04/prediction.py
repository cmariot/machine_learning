import numpy as np
from typing import Union


def predict_(
        x: np.ndarray, theta: np.ndarray) -> Union[np.ndarray, None]:

    """
        Computes the vector of prediction y_hat from two non-empty numpy.array.
        Args:
            x: has to be an numpy.array, a vector of dimension m * 1.
            theta: has to be an numpy.array, a vector of dimension 2 * 1.
        Returns:
            y_hat as a numpy.array, a vector of dimension m * 1.
            None if x and/or theta are not numpy.array. OK
            None if x or theta are empty numpy.array. OK
            None if x or theta dimensions are not appropriate.
        Raises:
            This function should not raise any Exceptions.
    """

    # Check if x and theta are non empty numpy.ndarray
    for arr in [x, theta]:
        if not isinstance(arr, np.ndarray):
            return None
        if arr.size == 0:
            return None

    # Check the dimension of theta
    if theta.size != 2 or theta.ndim != 1:
        return None

    # Check the dimension of x
    m = x.size
    if x.ndim != 1:
        return None

    # Add a column of 1's to x
    x_prime = np.c_[np.ones(m), x]

    # Compute y_hat, the vector of prediction as ndarray of float
    return np.matmul(x_prime, theta)


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
