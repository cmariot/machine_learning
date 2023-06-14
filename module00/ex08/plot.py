import numpy as np
import matplotlib.pyplot as plt
from typing import Union


def add_intercept(x) -> Union[np.ndarray, None]:
    """Adds a column of 1's to the non-empty numpy.array x. Args:
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
    if x.size == 0:
        return None
    return np.c_[np.ones(x.shape[0]), x]


def predict_(
        x: np.ndarray, theta: np.ndarray) -> Union[np.ndarray, None]:

    """
        Computes the vector of prediction y_hat from two non-empty numpy.array.
        Args:
            x: has to be an numpy.array, a vector of dimension m * 1.
            theta: has to be an numpy.array, a vector of dimension 2 * 1.
        Returns:
            y_hat as a numpy.array, a vector of dimension m * 1.
            None if x and/or theta are not numpy.array.
            None if x or theta are empty numpy.array.
            None if x or theta dimensions are not appropriate.
        Raises:
            This function should not raise any Exceptions.
    """

    # Check if theta is a numpy.ndarray
    if not isinstance(theta, np.ndarray):
        return None

    # Add a column of ones to the vector x
    X = add_intercept(x)

    # If x is not a numpy.ndarray or x is empty, return None
    if X is None:
        return None

    # Check the dimension of x and theta
    m = x.shape[0]
    if x.shape != (m, ) or theta.shape != (2, ):
        return None

    # Compute y_hat, the vector of prediction with a matrix multiplication
    y_hat = np.matmul(X, theta)

    return y_hat


def loss_elem_(y, y_hat):
    """
        Description:
            Calculates all the elements (y_pred - y)^2 of the
            loss function.
        Args:
            y: has to be an numpy.array, a vector.
            y_hat: has to be an numpy.array, a vector.
        Returns:
            J_elem: numpy.array, a vector of dimension
                (number of the training examples, 1).
            None if there is a dimension problem between X, Y or theta.
            None if any argument is not of the expected type.
        Raises:
            This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    m = y.shape[0]
    if m == 0:
        return None
    if y.shape != (m, ) or y_hat.shape != (m, ):
        return None
    return (y_hat - y) ** 2


def loss_(y, y_hat):
    """
    Computes the half mean squared error of two non-empty numpy.array,
    without any for loop.
    The two arrays must have the same dimensions.
    Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
    Returns:
        The half mean squared error of the two vectors as a float.
        None if y or y_hat are empty numpy.array.
        None if y and y_hat does not share the same dimensions.
    Raises:
        This function should not raise any Exceptions.
    """

    # Compute the loss by calling loss_elem_
    J_elem = loss_elem_(y, y_hat)
    if J_elem is None:
        return None

    # Compute the mean of J_elem
    J_value = np.mean(J_elem) / 2
    return J_value


def plot_with_loss(x, y, theta):
    """
    Plot the data and prediction line from three non-empty numpy.ndarray.
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * 1.
        y: has to be an numpy.ndarray, a vector of dimension m * 1.
        theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
        Nothing.
    Raises:
        This function should not raise any Exception.
    """

    # Check the input arguments
    if not all(isinstance(arr, np.ndarray) for arr in [x, y, theta]):
        return None

    if any(arr.size == 0 for arr in [x, y]):
        return None

    if any(arr.shape != (arr.size,) for arr in [x, y]) or theta.shape != (2,):
        return None

    # Predict y_hat
    y_hat = predict_(x, theta)

    if y_hat is None:
        return None

    # Plot the data
    plt.plot(x, y, 'o')

    # Plot the prediction line
    plt.plot(x, y_hat)

    # Plot the vertical dashed lines
    for i in range(x.shape[0]):
        plt.plot([x[i], x[i]], [y[i], y_hat[i]], 'r--')

    # Compute the value of the loss
    loss = loss_(y, y_hat) * 2

    # Add a title with a field-width of 4
    plt.title("Cost : {:4f}".format(loss))

    plt.show()


if __name__ == "__main__":

    x = np.arange(1, 6)
    y = np.array(
        [11.52434424, 10.62589482, 13.14755699, 18.60682298, 14.14329568])

    theta1 = np.array([18, -1])
    plot_with_loss(x, y, theta1)

    theta2 = np.array([14, 0])
    plot_with_loss(x, y, theta2)

    theta3 = np.array([12, 0.8])
    plot_with_loss(x, y, theta3)
