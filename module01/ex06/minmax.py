import numpy as np


def minmax(x: np.ndarray):
    """
    Computes the normalized version of a non-empty numpy.ndarray
    using the min-max standardization.

    Args:
        x: has to be an numpy.ndarray, a vector.

    Returns:
        x' as a numpy.ndarray.
        None if x is an empty numpy.ndarray or not a numpy.ndarray.

    Raises:
        This function shouldn't raise any Exception.

    """

    if not isinstance(x, np.ndarray):
        return None
    x_size = x.size
    if x_size == 0:
        return None
    if x.shape != (x_size, ):
        x = x.reshape((-1, ))
    if x.shape != (x_size, ):
        return None
    return (x - np.min(x)) / (np.max(x) - np.min(x))


if __name__ == "__main__":

    X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((-1, 1))
    print(minmax(X))
    # Output:
    # array([0.58333333, 1. , 0.33333333, 0.77777778, 0.91666667,
    #        0.66666667, 0.])

    # Example 2:
    Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    print(minmax(Y))
    # Output:
    # array([0.63636364, 1. , 0.18181818, 0.72727273, 0.93939394,
    #        0.6969697 , 0.])
