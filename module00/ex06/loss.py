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

    # Check if y and y_hat are non empty ndarray
    for arr in [y, y_hat]:
        if not isinstance(arr, np.ndarray):
            return None
        elif arr.size == 0:
            return None

    # Check the dimensions of y and y_hat
    m = y.size
    if y_hat.shape != (m, 1) or y.shape != (m, 1):
        return None

    # Compute J_elem, a vector
    J_elem = np.square(y_hat - y)
    return J_elem


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

    # Compute J_elem, if None : type or dimmension error
    J_elem = loss_elem_(y, y_hat)
    if J_elem is None:
        return None

    # Mean = sum / m
    J_value = np.mean(J_elem) / 2

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
