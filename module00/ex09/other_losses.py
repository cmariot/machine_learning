import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt


def mse_elem(y, y_hat) -> np.ndarray:
    return (y_hat - y) ** 2


def mse_(y, y_hat) -> float:
    """
        Description:
            Calculate the MSE between the predicted output and the real output.
        Args:
            y: has to be a numpy.array, a vector of dimension m * 1.
            y_hat: has to be a numpy.array, a vector of dimension m * 1.
        Returns:
            mse: has to be a float.
            None if there is a matching dimension problem.
        Raises:
            This function should not raise any Exceptions.
    """
    if any(not isinstance(_, np.ndarray) for _ in [y, y_hat]):
        return None

    m = y.shape[0]
    if m == 0 or y.shape != (m, ) or y_hat.shape != (m, ):
        return None

    if any(not np.all(np.isreal(_)) for _ in [y, y_hat]):
        return None

    J_elem = mse_elem(y, y_hat)

    return J_elem.mean()


def rmse_elem(y, y_hat):
    return (y_hat - y) ** 2


def rmse_(y, y_hat):
    """
        Description:
            Calculate the RMSE between the predicted output
            and the real output.
        Args:
            y: has to be a numpy.array, a vector of dimension m * 1.
            y_hat: has to be a numpy.array, a vector of dimension m * 1.
        Returns:
            rmse: has to be a float.
            None if there is a matching dimension problem.
        Raises:
            This function should not raise any Exceptions.
    """
    if any(not isinstance(_, np.ndarray) for _ in [y, y_hat]):
        return None

    m = y.shape[0]
    if m == 0 or y.shape != (m, ) or y_hat.shape != (m, ):
        return None

    if any(not np.all(np.isreal(_)) for _ in [y, y_hat]):
        return None

    J_elem = rmse_elem(y, y_hat).mean()

    return sqrt(J_elem)


def mae_elem(y_hat, y):
    return abs(y_hat - y)


def mae_(y, y_hat):
    """
        Description:
            Calculate the MAE between the predicted output and the real output.
        Args:
            y: has to be a numpy.array, a vector of dimension m * 1.
            y_hat: has to be a numpy.array, a vector of dimension m * 1.
        Returns:
            mae: has to be a float.
            None if there is a matching dimension problem.
        Raises:
            This function should not raise any Exceptions.
    """

    if any(not isinstance(_, np.ndarray) for _ in [y, y_hat]):
        return None

    m = y.shape[0]
    if m == 0 or y.shape != (m, ) or y_hat.shape != (m, ):
        return None

    if any(not np.all(np.isreal(_)) for _ in [y, y_hat]):
        return None

    J_elem = mae_elem(y, y_hat)
    return J_elem.mean()


def r2score_elem(y, y_hat):

    m = y.shape[0]
    mean = y.mean()

    numerator = 0
    denominator = 0
    for i in range(m):
        numerator += (y_hat[i] - y[i]) ** 2
        denominator += (y[i] - mean) ** 2

    return 1 - (numerator / denominator)


def r2score_(y, y_hat):
    """
        Description:
            Calculate the R2score between the predicted output and the output.
        Args:
            y: has to be a numpy.array, a vector of dimension m * 1.
            y_hat: has to be a numpy.array, a vector of dimension m * 1.
        Returns:
            r2score: has to be a float.
            None if there is a matching dimension problem.
        Raises:
            This function should not raise any Exceptions.
    """

    if any(not isinstance(_, np.ndarray) for _ in [y, y_hat]):
        return None

    m = y.shape[0]
    if m == 0 or y.shape != (m, ) or y_hat.shape != (m, ):
        return None

    if any(not np.all(np.isreal(_)) for _ in [y, y_hat]):
        return None

    r2_score_elem = r2score_elem(y, y_hat)
    return r2_score_elem


if __name__ == "__main__":

    x = np.array([0, 15, -9, 7, 12, 3, -21])
    y = np.array([2, 14, -13, 5, 12, 4, -19])

    # Mean squared error
    # your implementation
    print(mse_(x, y))
    # Output: 4.285714285714286
    # sklearn implementation
    print(mean_squared_error(x, y))
    # Output: 4.285714285714286

    # Root mean squared error
    # your implementation
    print(rmse_(x, y))
    # Output: 2.0701966780270626
    # sklearn implementation not available: take the square root of MSE
    print(sqrt(mean_squared_error(x, y)))
    # Output:
    # 2.0701966780270626

    # Mean absolute error
    # your implementation
    print(mae_(x, y))
    # Output: 1.7142857142857142
    # sklearn implementation
    print(mean_absolute_error(x, y))
    # Output:
    # 1.7142857142857142

    # R2-score
    # your implementation
    print(r2score_(x, y))
    # Output: 0.9681721733858745
    # sklearn implementation
    print(r2_score(x, y))
    # Output: 0.9681721733858745
