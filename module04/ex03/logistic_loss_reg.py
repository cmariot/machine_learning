import numpy as np
import sklearn.metrics as skm


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
        theta_prime = np.copy(theta)
        theta_prime[0, 0] = 0
        regularization = np.dot(theta_prime.T, theta_prime)
        return float(regularization[0, 0])

    except Exception:
        return None


def checkargs_reg_log_loss_(func):

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


def vec_log_loss_(y, y_hat, eps=1e-15):
    """
        Compute the logistic loss value.
        Args:
            y: has to be an numpy.ndarray, a vector of shape m * 1.
            y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
            eps: epsilon (default=1e-15)
        Returns:
            The logistic loss value as a float.
            None on any error.
        Raises:
            This function should not raise any Exception.
    """

    if not all(isinstance(x, np.ndarray) for x in [y, y_hat]):
        return None

    m = y.shape[0]
    n = y.shape[1]

    if (m == 0 or n == 0):
        return None
    elif y_hat.shape != (m, n):
        return None

    try:
        y_hat_clipped = np.clip(y_hat, eps, 1 - eps)
        const = -1.0 / m
        dot1 = np.dot(y.T, np.log(y_hat_clipped))
        dot2 = np.dot((1 - y).T, np.log(1 - y_hat_clipped))
        return (const * (dot1 + dot2)[0, 0])

    except Exception:
        return None


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
        loss = vec_log_loss_(y, y_hat)
        if loss is None:
            return None
        l2_ = l2(theta)
        if l2_ is None:
            return None
        reg = (lambda_ / (2 * y.shape[0])) * l2_
        return loss + reg
    except Exception:
        return None


if __name__ == "__main__":

    y = np.array([1, 1, 0, 0, 1, 1, 0]).reshape((-1, 1))
    y_hat = np.array([.9, .79, .12, .04, .89, .93, .01]).reshape((-1, 1))
    theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))

    # Example :
    print(reg_log_loss_(y, y_hat, theta, .5))
    reg_term = (.5 / (2 * y.shape[0])) * l2(theta)
    print(skm.log_loss(y, y_hat, eps=1e-15, labels=[0, 1]) + reg_term)
    # Output:
    # 0.43377043716475955

    # Example :
    print(reg_log_loss_(y, y_hat, theta, .05))
    reg_term = (.05 / (2 * y.shape[0])) * l2(theta)
    print(skm.log_loss(y, y_hat, eps=1e-15, labels=[0, 1]) + reg_term)
    # Output:
    # 0.13452043716475953

    # Example :
    print(reg_log_loss_(y, y_hat, theta, .9))
    reg_term = (.9 / (2 * y.shape[0])) * l2(theta)
    print(skm.log_loss(y, y_hat, eps=1e-15, labels=[0, 1]) + reg_term)
    # Output:
    # 0.6997704371647596
