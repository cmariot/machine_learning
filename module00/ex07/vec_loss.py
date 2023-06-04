import numpy as np


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
    if y.shape != (m, 1) or y_hat.shape != (m, 1):
        return None
    if np.isreal(y).all() is False or np.isreal(y_hat).all() is False:
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
    J_value = np.mean(J_elem)
    return J_value / 2


if __name__ == "__main__":
    X = np.array([[0], [15], [-9], [7], [12], [3], [-21]])
    Y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])

    # Example 1:
    print(loss_(X, Y))
    # Output:
    # 2.142857142857143

    # Example 2:
    print(loss_(X, X))
    # Output:
    # 0.0
