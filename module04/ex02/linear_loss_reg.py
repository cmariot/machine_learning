import numpy as np


def checkargs_l2_(func):

    def wrapper(theta):
        if not isinstance(theta, np.ndarray):
            return None
        elif theta.size == 0:
            return None
        return func(theta)

    return wrapper


@checkargs_l2_
def l2(theta):
    """
    Computes the L2 regularization of a non-empty numpy.ndarray,
    without any for-loop.
    Args:
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
    Returns:
        The L2 regularization as a float.
        None if theta in an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """

    try:
        theta_prime = theta.copy()
        theta_prime[0, 0] = 0
        regularization = np.dot(theta_prime.T, theta_prime)
        return float(regularization[0, 0])

    except Exception:
        return None


def checkargs_reg_loss_(func):

    def wrapper(y, y_hat, theta, lambda_):
        try:
            if not isinstance(y, np.ndarray) \
                or not isinstance(y_hat, np.ndarray) \
                    or not isinstance(theta, np.ndarray):
                return None
            m = y.shape[0]
            n = theta.shape[0]
            if m == 0 or n == 0:
                return None
            if y.shape != (m, 1) \
                or y_hat.shape != (m, 1) \
                    or theta.shape != (n, 1):
                return None
            if not isinstance(lambda_, float):
                return None
            return func(y, y_hat, theta, lambda_)
        except Exception:
            return None

    return wrapper


@checkargs_reg_loss_
def reg_loss_(y, y_hat, theta, lambda_):
    """
    Computes the regularized loss of a linear regression model
    from two non-empty numpy.array, without any for loop.
    Args:
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
        lambda_: has to be a float.
    Returns:
        The regularized loss as a float.
        None if y, y_hat, or theta are empty numpy.ndarray.
        None if y and y_hat do not share the same shapes.
    Raises:
        This function should not raise any Exception.
    """

    try:
        m = y.shape[0]
        const = 1 / (2 * m)
        dot = (y_hat - y).T.dot(y_hat - y)
        l2_ = l2(theta)
        if l2_ is None:
            return None
        reg = lambda_ * l2_
        return float(const * np.sum(dot + reg))
    except Exception:
        return None


if __name__ == "__main__":

    y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    y_hat = np.array([3, 13, -11.5, 5, 11, 5, -20]).reshape((-1, 1))
    theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))

    # Example :
    print(reg_loss_(y, y_hat, theta, .5))
    # Output:
    # 0.8503571428571429

    # Example :
    print(reg_loss_(y, y_hat, theta, .05))
    # Output:
    # 0.5511071428571429

    # Example :
    print(reg_loss_(y, y_hat, theta, .9))
    # Output:
    # 1.1163571428571428
