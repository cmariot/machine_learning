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

    # Check if x and theta are numpy.ndarray
    if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
        return None

    # Check the dimension of x and theta
    m = x.shape[0]
    if m == 0:
        return None
    elif x.shape != (m, ) or theta.shape != (2, ):
        return None

    # Check if x and theta are float or int array
    if np.isreal(x).all() is False or np.isreal(theta).all() is False:
        return None

    # Create a numpy.ndarray of prediction
    y_hat = np.ndarray(m)

    # Compute y_hat, the vector of prediction
    # y_hat = theta[0] + theta[1] * x
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
