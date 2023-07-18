import numpy as np


def checkargs_reg_linear_grad_(func):

    def wrapper(y, x, theta, lambda_):
        try:
            if not isinstance(y, np.ndarray) \
                or not isinstance(x, np.ndarray) \
                    or not isinstance(theta, np.ndarray):
                return None
            m, n = x.shape
            if m == 0 or n == 0:
                return None
            if y.shape != (m, 1) \
                or x.shape != (m, n) \
                    or theta.shape != (n + 1, 1):
                return None
            if not isinstance(lambda_, (int, float)):
                return None
            return func(y, x, theta, lambda_)
        except Exception:
            return None

    return wrapper


@checkargs_reg_linear_grad_
def reg_linear_grad(y, x, theta, lambda_):
    """
    Computes the regularized linear gradient of three non-empty numpy.ndarray,
    with two for-loop. The three arrays must have compatible shapes.
    Args:
        y: has to be a numpy.ndarray, a vector of shape m * 1.
        x: has to be a numpy.ndarray, a matrix of dimesion m * n.
        theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
        lambda_: has to be a float.
    Return:
        A numpy.ndarray, a vector of shape (n + 1) * 1,
          containing the results of the formula for all j.
        None if y, x, or theta are empty numpy.ndarray.
        None if y, x or theta does not share compatibles shapes.
        None if y, x or theta or lambda_ is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    try:
        m, n = x.shape
        X_prime = np.hstack((np.ones((m, 1)), x))
        y_hat = X_prime.dot(theta)
        diff = y_hat - y
        theta_prime = theta.copy()
        theta_prime[0, 0] = 0.0
        gradient = np.zeros((n + 1, 1))
        gradient[0, 0] = np.sum(diff)
        for j in range(1, n + 1):
            for i in range(m):
                gradient[j, 0] += diff[i, 0] * x[i, j - 1]
            gradient[j, 0] += lambda_ * theta_prime[j, 0]
        return gradient / m

    except Exception:
        return None


@checkargs_reg_linear_grad_
def vec_reg_linear_grad(y, x, theta, lambda_):
    """
    Computes the regularized linear gradient of three non-empty numpy.ndarray,
    without any for-loop. The three arrays must have compatible shapes.
    Args:
        y: has to be a numpy.ndarray, a vector of shape m * 1.
        x: has to be a numpy.ndarray, a matrix of dimesion m * n.
        theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
        lambda_: has to be a float.
    Return:
        A numpy.ndarray, a vector of shape (n + 1) * 1,
          containing the results of the formula for all j.
        None if y, x, or theta are empty numpy.ndarray.
        None if y, x or theta does not share compatibles shapes.
        None if y, x or theta or lambda_ is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    try:
        m = y.shape[0]
        theta_prime = theta.copy()
        theta_prime[0, 0] = 0.0
        X_prime = np.hstack((np.ones((m, 1)), x))
        y_hat = X_prime.dot(theta)
        return (np.dot(X_prime.T, y_hat - y)
                + (lambda_ * theta_prime)) / m

    except Exception:
        return None


if __name__ == "__main__":

    x = np.array([[-6, -7, -9],
                  [13, -2, 14],
                  [-7, 14, -1],
                  [-8, -4, 6],
                  [-5, -9, 6],
                  [1, -5, 11],
                  [9, -11, 8]])
    y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
    theta = np.array([[7.01], [3], [10.5], [-6]])

    # Example 1.1:
    print(reg_linear_grad(y, x, theta, 1))
    # Output:
    # array([[ -60.99 ],
    #        [-195.64714286],
    #        [ 863.46571429],
    #        [-644.52142857]])

    # Example 1.2:
    print(vec_reg_linear_grad(y, x, theta, 1))
    # Output:
    # array([[ -60.99 ],
    #        [-195.64714286],
    #        [ 863.46571429],
    #        [-644.52142857]])

    # Example 2.1:
    print(reg_linear_grad(y, x, theta, 0.5))
    # Output:
    # array([[ -60.99 ],
    #        [-195.86142857],
    #        [ 862.71571429],
    #        [-644.09285714]])

    # Example 2.2:
    print(vec_reg_linear_grad(y, x, theta, 0.5))
    # Output:
    # array([[ -60.99 ],
    #        [-195.86142857],
    #        [ 862.71571429],
    #        [-644.09285714]])

    # Example 3.1:
    print(reg_linear_grad(y, x, theta, 0.0))
    # Output:
    # array([[ -60.99 ],
    #        [-196.07571429],
    #        [ 861.96571429],
    #        [-643.66428571]])

    # Example 3.2:
    print(vec_reg_linear_grad(y, x, theta, 0.0))
    # Output:
    # array([[ -60.99 ],
    #        [-196.07571429],
    #        [ 861.96571429],
    #        [-643.66428571]])
