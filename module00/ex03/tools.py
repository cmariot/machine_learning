import numpy as np
from typing import Union


def add_intercept(x) -> Union[np.ndarray, None]:
    """
        Adds a column of 1's to the non-empty numpy.array x.
        Used to perform a linear algebra trick for matrix multiplication.
        Args:
          x: has to be a numpy.array of dimension m * n.
        Returns:
          X, a numpy.array of dimension m * (n + 1).
          None if x is not a numpy.array.
          None if x is an empty numpy.array.
        Raises:
          This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray):
        return None

    m = x.shape[0]
    n = x.shape[1] if x.ndim > 1 else 1

    if m == 0 or n == 0:
        return None

    return np.c_[np.ones(m), x]


if __name__ == "__main__":

    x = np.arange(1, 6)
    X = add_intercept(x)
    print(X)

    # Output:
    # array([[1., 1.],
    #        [1., 2.],
    #        [1., 3.],
    #        [1., 4.],
    #        [1., 5.]])

    y = np.arange(1, 10).reshape((3, 3))
    Y = add_intercept(y)
    print(Y)

    # Output:
    # array([[1., 1., 2., 3.],
    #        [1., 4., 5., 6.],
    #        [1., 7., 8., 9.]])
