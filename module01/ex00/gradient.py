import numpy as np


def simple_gradient(x: np.ndarray,
                    y: np.ndarray,
                    theta: np.ndarray) -> np.ndarray:
    """
    Computes a gradient vector from 3 non-empty numpy.array,
    with a for-loop.
    The three arrays must have compatible shapes.
    Args:
        x: has to be an numpy.array, a vector of shape m * 1.
        y: has to be an numpy.array, a vector of shape m * 1.
        theta: has to be an numpy.array, a 2 * 1 vector.
    Return:
        The gradient as a numpy.array, a vector of shape 2 * 1.
        None if x, y, or theta are empty numpy.array.
        None if x, y and theta do not have compatible shapes.
        None if x, y or theta is not of the expected type.
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

    # Compute gradient
    gradient = np.zeros((2, 1))
    for i in range(x_size):

        # Compute predicted value
        y_hat = theta[0] + theta[1] * x[i]

        # Compute difference between predicted and real value
        diff = y_hat - y[i]

        # Update gradient
        gradient[0] += diff
        gradient[1] += diff * x[i]

    return gradient / x_size


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
    # array([[-19.0342574], [-586.66875564]])

    # Example 1:
    theta2 = np.array([1, -0.4]).reshape((-1, 1))
    gradient = simple_gradient(x, y, theta2)
    print(gradient)
    # Output:
    # array([[-57.86823748], [-2230.12297889]])
