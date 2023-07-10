import numpy as np


def add_polynomial_features(x, power):
    """
    Add polynomial features to matrix x by raising its columns
    to every power in the range of 1 up to the power given in argument.
    Args:
        x: has to be an numpy.ndarray, a matrix of shape m * n.
        power: has to be an int, the power up to which the columns
                of matrix x are going to be raised.
    Returns:
        The matrix of polynomial features as a numpy.ndarray,
            of shape m * (np), containg the polynomial feature values
            for all training examples.
        None if x is an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """

    try:
        if not isinstance(x, np.ndarray):
            return None
        m, n = x.shape
        if m == 0 or n == 0:
            return None
        if not isinstance(power, int) or power < 1:
            return None
        polynomial_matrix = x
        for i in range(2, power + 1):
            new_column = x ** i
            polynomial_matrix = np.c_[polynomial_matrix, new_column]
        return polynomial_matrix

    except Exception:
        return None


if __name__ == "__main__":

    x = np.arange(1, 11).reshape(5, 2)
    print(x)

    polynomial_matrix = add_polynomial_features(x, 3)
    print(polynomial_matrix)
    # Output:
    # array([[   1,    2,    1,    4,    1,    8],
    #        [   3,    4,    9,   16,   27,   64],
    #        [   5,    6,   25,   36,  125,  216],
    #        [   7,    8,   49,   64,  343,  512],
    #        [   9,   10,   81,  100,  729, 1000]])

    polynomial_matrix = add_polynomial_features(x, 4)
    print(polynomial_matrix)
    # Output:
    # array([[    1,     2,     1,     4,     1,     8,     1,    16],
    #        [    3,     4,     9,    16,    27,    64,    81,   256],
    #        [    5,     6,    25,    36,   125,   216,   625,  1296],
    #        [    7,     8,    49,    64,   343,   512,  2401,  4096],
    #        [    9,    10,    81,   100,   729,  1000,  6561, 10000]])
