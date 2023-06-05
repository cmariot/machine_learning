import numpy as np
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
    if x.shape != (m, 1) or theta.shape != (2, 1):
        return None

    # Compute y_hat, the vector of prediction with a matrix multiplication
    y_hat = np.matmul(X, theta)

    return y_hat


def loss_elem_(y: np.ndarray, y_hat: np.ndarray) -> Union[np.ndarray, None]:
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
    for arr in [y, y_hat]:
        if not isinstance(arr, np.ndarray):
            return None
    m = y.shape[0]
    if m == 0:
        return None
    if y.shape != (m, 1) or y_hat.shape != (m, 1):
        return None

    return (y_hat - y) ** 2


def loss_(y, y_hat):
    """
        Description:
            Calculates the value of loss function.
        Args:
            y: has to be an numpy.array, a vector.
            y_hat: has to be an numpy.array, a vector.
        Returns:
            J_value : has to be a float.
            None if there is a dimension problem between X, Y or theta.
            None if any argument is not of the expected type.
        Raises:
            This function should not raise any Exception.
    """
    loss_elem = loss_elem_(y, y_hat)
    if loss_elem is None:
        return None
    J_value = np.mean(loss_elem) / 2
    return J_value


if __name__ == "__main__":

    # Example 1:
    # x1 is a vector of dimension 5
    x1 = np.array([[0.], [1.], [2.], [3.], [4.]])
    # theta1 is a vector of dimension 2, the equation is y = 2 * x + 4
    theta1 = np.array([[2.], [4.]])
    # y_hat is a vector of dimension 5, the predict value of each x1 point with
    # the equation y = 2 * x + 4
    y_hat1 = predict_(x1, theta1)

    # y1 is a vector of dimension 5, the real value of each x1 point
    y1 = np.array([[2.], [7.], [12.], [17.], [22.]])

    # Example 1:
    loss_elem = loss_elem_(y1, y_hat1)
    print(loss_elem)
    # Output:
    # array([[0.], [1], [4], [9], [16]])

    # Example 2:
    loss = loss_(y1, y_hat1)
    print(loss)
    # Output:
    # 3.0

    x2 = np.array([0, 15, -9, 7, 12, 3, -21]).reshape(-1, 1)
    theta2 = np.array([[0.], [1.]]).reshape(-1, 1)
    y_hat2 = predict_(x2, theta2)
    y2 = np.array([2, 14, -13, 5, 12, 4, -19]).reshape(-1, 1)

    # Example 3:
    loss = loss_(y2, y_hat2)
    print(loss)
    # Output:
    # 2.142857142857143

    # Example 4:
    loss = loss_(y2, y2)
    print(loss)
    # Output:
    # 0.0
