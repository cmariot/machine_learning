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

    for arr in [x, theta]:
        if not isinstance(arr, np.ndarray):
            return None
        elif arr.size == 0:
            return None

    m = x.shape[0]
    n = x.shape[1]

    if theta.shape != (n + 1, 1):
        return None

    # Add a column of 1 to x -> X_prime
    X_prime = np.concatenate((np.ones((m, 1)), x), axis=1)

    y_hat = np.dot(X_prime, theta)
    return y_hat


def gradient_(x, y, theta):
    """
    Computes a gradient vector from three non-empty numpy.array,
    without any for-loop.
    The three arrays must have the compatible dimensions.
    Args:
        x: has to be an numpy.array, a matrix of dimension m * n.
        y: has to be an numpy.array, a vector of dimension m * 1.
        theta: has to be an numpy.array, a vector (n + 1) * 1.
    Return:
        The gradient as a numpy.array, a vector of dimensions n * 1,
            containg the result of the formula for all j.
        None if x, y, or theta are empty numpy.array.
        None if x, y and theta do not have compatible dimensions.
        None if x, y or theta is not of expected type.
    Raises:
        This function should not raise any Exception.
    """

    for array in [x, y, theta]:
        if not isinstance(array, np.ndarray):
            return None
    m, n = x.shape
    if m == 0 or n == 0:
        return None
    elif y.shape != (m, 1):
        return None
    elif theta.shape != (n + 1, 1):
        return None
    X_prime = np.c_[np.ones(m), x]
    return (1 / m) * (X_prime.T.dot(X_prime.dot(theta) - y))


def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
        Fits the model to the training dataset contained in x and y.
    Args:
        x: has to be a numpy.array, a matrix of dimension m * n:
                    (number of training examples, number of features).
        y: has to be a numpy.array, a vector of dimension m * 1:
                    (number of training examples, 1).
        theta: has to be a numpy.array, a vector of dimension (n + 1) * 1:
                    (number of features + 1, 1).
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done during
        the gradient descent
    Return:
        new_theta: np.array, a vector of dimension (number of features + 1, 1).
        None if there is a matching dimension problem.
        None if x, y, theta, alpha or max_iter is not of expected type.
    Raises:
        This function should not raise any Exception.
    """

    # Check the arguments type
    for arr in [x, y, theta]:
        if not isinstance(arr, np.ndarray):
            return None
    if not isinstance(alpha, float) or alpha < 0.:
        return None
    elif not isinstance(max_iter, int) or max_iter < 0:
        return None

    # Check the arguments shape
    m, n = x.shape
    if m == 0 or n == 0:
        return None
    if y.shape != (m, 1):
        return None
    elif theta.shape != ((n + 1), 1):
        return None

    # Train the model to fit the data
    for _ in range(max_iter):
        gradient = gradient_(x, y, theta)
        if gradient is None:
            return None
        if all(__ == 0. for __ in gradient):
            break
        theta = theta - alpha * gradient
    return theta


if __name__ == "__main__":

    x = np.array([[0.2, 2., 20.],
                  [0.4, 4., 40.],
                  [0.6, 6., 60.],
                  [0.8, 8., 80.]])

    y = np.array([[19.6],
                  [-2.8],
                  [-25.2],
                  [-47.6]])

    theta = np.array([[42.],
                      [1.],
                      [1.],
                      [1.]])

    # Example 0:
    theta2 = fit_(x, y, theta, alpha=0.0005, max_iter=42000)
    print(theta2)
    # Output:
    # array([[41.99..],[0.97..], [0.77..], [-1.20..]])

    # Example 1:
    y_hat = predict_(x, theta2)
    print(y_hat)
    # Output:
    # array([[19.5992..], [-2.8003..], [-25.1999..], [-47.5996..]])
