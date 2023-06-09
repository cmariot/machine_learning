import numpy as np


def predict_(
        x: np.ndarray, theta: np.ndarray):

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
    if theta.shape != (2, 1):
        return None

    # Check the dimension of x
    m = x.size
    if x.shape != (m, 1):
        return None

    # Add a column of 1's to x
    x_prime = np.c_[np.ones(m), x]

    # Compute y_hat, the vector of prediction as ndarray of float
    return x_prime @ theta


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
    if not isinstance(alpha, float) or alpha <= 0.0:
        return None
    elif not isinstance(max_iter, int) or max_iter <= 0:
        return None

    # Gradient descent
    gradient = np.zeros((2, 1))
    for _ in range(max_iter):
        gradient = simple_gradient(x, y, theta)
        if gradient is None:
            return None
        elif all(val == [0.0] for val in gradient):
            break
        theta = theta - alpha * gradient
        print(" {:2.2f} %" .format(_ / max_iter * 100), end="\r")
    return theta


if __name__ == "__main__":
    x = np.array(
        [[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
    y = np.array(
        [[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
    theta = np.array([1, 1]).reshape((-1, 1))

    # Example 0:
    theta1 = fit_(x, y, theta, alpha=5e-8, max_iter=1_500_000)
    print(theta1)
    # Output:
    # array([[1.40709365],
    #        [1.1150909 ]])

    # Example 1:
    y_hat = predict_(x, theta1)
    print(y_hat)
    # Output:
    # array([[15.3408728 ],
    #        [25.38243697],
    #        [36.59126492],
    #        [55.95130097],
    #        [65.53471499]])
