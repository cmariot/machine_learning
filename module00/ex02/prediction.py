import numpy as np


def simple_predict(x: np.ndarray, theta: np.ndarray) -> np.ndarray:

    """
    Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
    Args:
      x: has to be an numpy.ndarray, a vector of dimension m * 1.
      theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
      y_hat as a numpy.ndarray, a vector of dimension m * 1.
      None if x or theta are empty numpy.ndarray.
      None if x or theta dimensions are not appropriate.
    Raises:
      This function should not raise any Exception
    """

    # Check if x and theta are numpy.ndarray, if they are not empty and
    # if they are array of numbers
    for arr in [x, theta]:
        if not isinstance(arr, np.ndarray):
            return None
        if arr.size == 0:
            return None

    # Check the dimension of theta
    if theta.size != 2 or theta.ndim != 1:
        return None

    # Check the dimension of x
    if x.ndim != 1:
        return None

    # Compute y_hat, the vector of prediction as ndarray of float
    m = x.size
    y_hat = np.zeros(m)
    for i in range(m):
        y_hat[i] = theta[0] + theta[1] * x[i]
    return y_hat


if __name__ == "__main__":

    x = np.arange(1, 6)
    # x = [1, 2, 3, 4, 5]

    theta1 = np.array([5, 0])
    y_hat = simple_predict(x, theta1)
    print(y_hat)
    # array([5., 5., 5., 5., 5.])

    theta2 = np.array([0, 1])
    y_hat = simple_predict(x, theta2)
    print(y_hat)
    # array([1., 2., 3., 4., 5.])

    theta3 = np.array([5, 3])
    y_hat = simple_predict(x, theta3)
    print(y_hat)
    # array([ 8., 11., 14., 17., 20.])

    theta4 = np.array([-3, 1])
    y_hat = simple_predict(x, theta4)
    print(y_hat)
    # array([-2., -1.,  0.,  1.,  2.])
