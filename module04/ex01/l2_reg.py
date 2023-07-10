import numpy as np


def checkargs_l2_(func):

    def wrapper(theta):
        if not isinstance(theta, np.ndarray):
            return None
        elif theta.size == 0:
            return None
        return func(theta)

    return wrapper


@checkargs_l2_
def iterative_l2(theta):
    """
    Computes the L2 regularization of a non-empty numpy.ndarray,
    with a for-loop.
    Args:
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
    Returns:
        The L2 regularization as a float.
        None if theta in an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """

    try:
        regularization = 0.0
        for i in range(1, theta.shape[0]):
            regularization += theta[i, 0] ** 2
        return regularization

    except Exception:
        return None


@checkargs_l2_
def l2(theta):
    """
    Computes the L2 regularization of a non-empty numpy.ndarray,
    without any for-loop.
    Args:
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
    Returns:
        The L2 regularization as a float.
        None if theta in an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """

    try:
        theta[0, 0] = 0
        regularization = np.dot(theta.T, theta)
        return float(regularization[0, 0])

    except Exception:
        return None


if __name__ == "__main__":

    x = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))

    # Example 1:
    print(iterative_l2(x))
    # Output:
    # 911.0

    # Example 2:
    print(l2(x))
    # Output:
    # 911.0

    y = np.array([3, 0.5, -6]).reshape((-1, 1))

    # Example 3:
    print(iterative_l2(y))
    # Output:
    # 36.25

    # Example 4:
    print(l2(y))
    # Output:
    # 36.25
