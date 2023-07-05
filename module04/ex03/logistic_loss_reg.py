import numpy as np


def checkargs_reg_log_loss_(func):

    def wrapper(y, y_hat, theta, lambda_):
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

    return wrapper


@checkargs_reg_log_loss_
def reg_log_loss_(y, y_hat, theta, lambda_):
    """
    Computes the regularized loss of a logistic regression model
    from two non-empty numpy.ndarray, without any for loop.
    Args:
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
        lambda_: has to be a float.
    Returns:
        The regularized loss as a float.
        None if y, y_hat, or theta is empty numpy.ndarray.
        None if y and y_hat do not share the same shapes.
    Raises:
        This function should not raise any Exception.
    """

    try:
        m = y.shape[0]

        loss = -1 / m * (y.T.dot(np.log(y_hat))
                         + (1 - y).T.dot(np.log(1 - y_hat))) \
                            + ((lambda_ / (2 * m)) * (theta[1:].T @ theta[1:]))
        return loss[0][0]

    except Exception:
        return None


if __name__ == "__main__":

    y = np.array([1, 1, 0, 0, 1, 1, 0]).reshape((-1, 1))
    y_hat = np.array([.9, .79, .12, .04, .89, .93, .01]).reshape((-1, 1))
    theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))

    # Example :
    print(reg_log_loss_(y, y_hat, theta, .5))
    # Output:
    # 0.43377043716475955

    # Example :
    print(reg_log_loss_(y, y_hat, theta, .05))
    # Output:
    # 0.13452043716475953

    # Example :
    print(reg_log_loss_(y, y_hat, theta, .9))
    # Output:
    # 0.6997704371647596
