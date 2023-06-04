import numpy as np
import matplotlib.pyplot as plt


def plot(x, y, theta):

    """
    Plot the data and prediction line from three non-empty numpy.array.
    Args:
        x: has to be an numpy.array, a vector of dimension m * 1.
        y: has to be an numpy.array, a vector of dimension m * 1.
        theta: has to be an numpy.array, a vector of dimension 2 * 1.
    Returns:
        Nothing.
    Raises:
        This function should not raise any Exceptions.
    """

    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) \
            or not isinstance(theta, np.ndarray):
        print("x, y and theta must be numpy.ndarray")
        return None
    elif x.size == 0 or y.size == 0 or theta.size == 0:
        print("x, y and theta must be non-empty numpy.ndarray")
        return None
    elif x.shape != (x.size, ) or y.shape != (y.size, ) \
            or theta.shape != (2, 1):
        print("x, y and theta must be numpy.ndarray of dimension 1")
        return None
    elif np.isreal(x).all() is False or np.isreal(y).all() is False \
            or np.isreal(theta).all() is False:
        print("x, y and theta must be numpy.ndarray of float or int")
        return None

    # Plot the data and prediction line
    try:
        plt.plot(x, y, 'o')
        plt.plot(x, theta[0] + theta[1] * x, '-')
        plt.show()
    except Exception as e:
        print(e)


if __name__ == "__main__":
    x = np.arange(1, 6)
    y = np.array([3.74013816, 3.61473236, 4.57655287, 4.66793434, 5.95585554])
    # Example 1:
    theta1 = np.array([[4.5], [-0.2]])
    plot(x, y, theta1)

    theta2 = np.array([[-1.5], [2]])
    plot(x, y, theta2)

    theta3 = np.array([[3], [0.3]])
    plot(x, y, theta3)
