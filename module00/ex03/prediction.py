import numpy as np


def simple_predict(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
    Args:
      x: has to be an numpy.ndarray, a matrix of dimension m * n.
      theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
    Returns:
      y_hat as a numpy.ndarray, a vector of dimension m * 1.
      None if x or theta are empty numpy.ndarray.
      None if x or theta dimensions are not appropriate.
    Raises:
      This function should not raise any Exception
    """
    # Check if x and theta are numpy.ndarray
    if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
        return None

    # Check the dimension of x and theta
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] + 1 != theta.size:
        return None

    # Check if x and theta are float or int array
    if not np.issubdtype(x.dtype, np.number) or not np.issubdtype(theta.dtype, np.number):
        return None

    # Add a column of ones to x
    x_with_ones = np.insert(x, 0, 1, axis=1)

    # Reshape theta to a column vector
    theta = theta.reshape(-1, 1)

    # Compute y_hat, the vector of prediction
    y_hat = x_with_ones.dot(theta)

    # Reshape y_hat to a vector
    y_hat = y_hat.reshape(-1)
    return y_hat


if __name__ == "__main__":

    x = np.ndarray((5, 1))
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
