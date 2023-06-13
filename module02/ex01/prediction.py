import numpy as np


def predict_(x, theta):
    """
    Computes the prediction vector y_hat from two non-empty numpy.array.
    Args:
        x: has to be an numpy.array, a vector of dimensions m * n.
        theta: has to be an numpy.array, a vector of dimensions (n + 1) * 1.
    Return:
        y_hat as a numpy.array, a vector of dimensions m * 1.
        None if x or theta are empty numpy.array.
        None if x or theta dimensions are not appropriate.
        None if x or theta is not of expected type.
    Raises:
        This function should not raise any Exception.
    """

    if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
        return None

    m = x.shape[0]
    n = x.shape[1]

    if m == 0 or n == 0 or theta.shape[0] != n + 1 or theta.shape[1] != 1:
        return None

    X = np.concatenate((np.ones((m, 1)), x), axis=1)
    y_hat = np.dot(X, theta)
    return y_hat


if __name__ == "__main__":

    x = np.arange(1, 13).reshape((4, -1))
    print(x)

    # Example 1:
    theta1 = np.array([5, 0, 0, 0]).reshape((-1, 1))
    print(predict_(x, theta1))
    # Ouput:
    # array([[5.], [5.], [5.], [5.]])
    # Do you understand why y_hat contains only 5’s here?

    # Example 2:
    theta2 = np.array([0, 1, 0, 0]).reshape((-1, 1))
    print(predict_(x, theta2))
    # Output:
    # array([[ 1.], [ 4.], [ 7.], [10.]])
    # Do you understand why y_hat == x[:,0] here?

    # Example 3:
    theta3 = np.array([-1.5, 0.6, 2.3, 1.98]).reshape((-1, 1))
    print(predict_(x, theta3))
    # Output:
    # array([[ 9.64], [24.28], [38.92], [53.56]])

    # Example 4:
    theta4 = np.array([-3, 1, 2, 3.5]).reshape((-1, 1))
    print(predict_(x, theta4))
    # Output:
    # array([[12.5], [32. ], [51.5], [71. ]])
