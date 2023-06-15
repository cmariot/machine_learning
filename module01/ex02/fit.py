import numpy as np
from typing import Union


def predict_(
        x: np.ndarray, theta: np.ndarray) -> Union[np.ndarray, None]:

    """
        Computes the vector of prediction y_hat from two non-empty numpy.array.
        Args:
            x: has to be an numpy.array, a vector of dimension m * 1.
            theta: has to be an numpy.array, a vector of dimension 2 * 1.
        Returns:
            y_hat as a numpy.array, a vector of dimension m * 1.
            None if x and/or theta are not numpy.array. OK
            None if x or theta are empty numpy.array. OK
            None if x or theta dimensions are not appropriate.
        Raises:
            This function should not raise any Exceptions.
    """

    # Check if x and theta are non empty numpy.ndarray
    for arr in [x, theta]:
        if not isinstance(arr, np.ndarray):
            return None
        if arr.size == 0:
            return None

    # Check the dimension of theta
    if theta.shape != (2, 1) and theta.shape != (1, 2):
        return None

    # Check the dimension of x
    if x.shape[0] != 1 and x.shape[1] != 1:
        return None

    # If x and theta are row vectors, reshape them as column vectors
    if theta.shape[0] == 1:
        theta = theta.reshape(-1, 1)
        if x.shape[0] == 1:
            x = x.reshape(-1, 1)
        else:
            return None

    # Size of the training set
    m = x.shape[0]

    # Add a column of 1's to x
    x_prime = np.c_[np.ones(m), x]

    # Compute y_hat, the vector of prediction as ndarray of float
    y_hat = np.matmul(x_prime, theta)

    return y_hat


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
    m = x.size
    if x.shape != (m, 1) or y.shape != (m, 1):
        return None

    # Check the shape of theta
    if theta.shape != (2, 1):
        return None

    # Matrix of shape m * 2
    Xprime = np.c_[np.ones((m, 1)), x]

    # Matrix of shape 2 * m
    XprimeT = Xprime.T

    gradient = np.matmul((XprimeT), (Xprime.dot(theta) - y)) / m
    return gradient


def fit_(x, y, theta, alpha, max_iter):
    """
         Description:
            Fits the model to the training dataset contained in x and y.
        Args:
            x: has to be a numpy.ndarray, a vector of dimension m * 1:
                (number of training examples, 1).
            y: has to be a numpy.ndarray, a vector of dimension m * 1:
                (number of training examples, 1).
            theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
            alpha: has to be a float, the learning rate
            max_iter: has to be an int, the number of iterations done
                during the gradient descent
         Returns:
            new_theta: numpy.ndarray, a vector of dimension 2 * 1.
            None if there is a matching dimension problem.
         Raises:
            This function should not raise any Exception.
    """

    # Check if x, y, and theta are non empty numpy arrays
    for arr in [x, y, theta]:
        if not isinstance(arr, np.ndarray):
            return None
        if arr.size == 0:
            return None

    # Check if x, y and theta have compatible shapes
    m = x.shape[0]
    if x.shape != (m, 1) or y.shape != (m, 1):
        return None
    elif theta.shape != (2, 1):
        return None

    # Check if alpha and max_iter types and if they are positive
    if not isinstance(alpha, float) or not isinstance(max_iter, int):
        return None
    if alpha <= 0 or max_iter <= 0:
        return None

    # Gradient descent
    gradient = np.zeros((2, 1))
    for _ in range(max_iter):
        gradient = simple_gradient(x, y, theta)
        if gradient is None:
            return None
        elif gradient[0] == 0. and gradient[1] == 0.:
            break
        theta = theta - alpha * gradient
    return theta


if __name__ == "__main__":
    x = np.array(
        [[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
    y = np.array(
        [[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
    theta = np.array([1, 1]).reshape((-1, 1))

    # Example 0:
    theta1 = fit_(x, y, theta, alpha=5e-8, max_iter=1500000)
    print(theta1)
    # Output:
    # array([[1.40709365],
    #        [1.1150909 ]])

    # Example 1:
    predict = predict_(x, theta1)
    print(predict)
    # Output:
    # array([[15.3408728 ],
    #        [25.38243697],
    #        [36.59126492],
    #        [55.95130097],
    #        [65.53471499]])
