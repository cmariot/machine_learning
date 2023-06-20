import numpy as np


def simple_gradient(x, y, theta):

    """
    Computes a gradient vector from 3 non-empty numpy.array,
    without any for loop.
    The three arrays must have compatible shapes.
    Args:
        x: has to be a numpy.array, a matrix of shape m * 1.
        y: has to be a numpy.array, a vector of shape m * 1.
        theta: has to be a numpy.array, a 2 * 1 vector.
    Return:
        The gradient as a numpy.ndarray, a vector of dimension 2 * 1.
        None if x, y, or theta is an empty numpy.ndarray.
        None if x, y and theta do not have compatible dimensions.
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
    x_size = x.size
    if x.shape != (x_size, 1) or y.shape != (x_size, 1):
        return None

    # Check the shape of theta
    if theta.shape != (2, 1):
        return None

    # Add a column of 1 -> Matrix of shape m * 2
    Xprime = np.c_[np.ones(x_size), x]

    # Compute and return the gradient
    gradient = Xprime.T @ (Xprime @ theta - y) / x_size
    return gradient


if __name__ == "__main__":

    x = np.array(
        [12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733]
        ).reshape((-1, 1))

    y = np.array(
        [37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554]
        ).reshape((-1, 1))

    # Example 0:
    theta1 = np.array([2, 0.7]).reshape((-1, 1))
    gradient = simple_gradient(x, y, theta1)
    print(gradient)
    # Output:
    # array([[-19.0342...], [-586.6687...]])

    # Example 1:
    theta2 = np.array([1, -0.4]).reshape((-1, 1))
    gradient = simple_gradient(x, y, theta2)
    print(gradient)
    # Output:
    # array([[-57.8682...], [-2230.1229...]])
