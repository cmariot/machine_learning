import numpy as np
import time
import shutil


def predict_(x, theta):
    for arr in [x, theta]:
        if not isinstance(arr, np.ndarray):
            return None
        elif arr.size == 0:
            return None
    m = x.shape[0]
    n = x.shape[1]
    if theta.shape != (n + 1, 1):
        return None
    X_prime = np.concatenate((np.ones((m, 1)), x), axis=1)
    return X_prime @ theta


def gradient_(x, y, theta):
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
    return (X_prime.T @ (X_prime @ theta - y)) / m


def ft_progress(iterable,
                length=shutil.get_terminal_size().columns - 2,
                fill='█',
                empty='░',
                print_end='\r'):
    """
    Progress bar generator
    This function displays a progress bar for an iterable object
    and yields its items.

    Parameters:
    - iterable (list): An iterable object.
    - length (int, optional): The length of the progress bar. Default is 50.
    - fill (str, optional): The character used to fill the progress bar.
    - print_end (str, optional): The character used to separate printed output.

    Returns:
    - generator: A generator that yields the items of the iterable object.

    Example usage:
    ```
        for i in ft_progress(range(100)):
            time.sleep(0.1)
    ```
    """

    total = len(iterable)
    start = time.time()
    for i, item in enumerate(iterable, start=1):
        elapsed_time = time.time() - start
        eta = elapsed_time * (total / i - 1)
        current_percent = (i / total) * 100
        filled_length = int(length * i / total)
        if eta == 0.0:
            eta_str = '[DONE]    '
        elif eta < 60:
            eta_str = f'[ETA {eta:.0f} s]'
        elif eta < 3600:
            eta_str = f'[ETA {eta / 60:.0f} m]'
        else:
            eta_str = f'[ETA {eta / 3600:.0f} h]'
        percent_str = f'[{current_percent:6.2f} %] '
        progress_str = fill * filled_length + empty * (length - filled_length)
        counter_str = f' [{i:>{len(str(total))}}/{total}] '
        if elapsed_time < 60:
            elapsed_time_str = f' [Elapsed-time {elapsed_time:.2f} s]'
        elif elapsed_time < 3600:
            elapsed_time_str = f' [Elapsed-time {elapsed_time / 60:.2f} m]'
        else:
            elapsed_time_str = f' [Elapsed-time {elapsed_time / 3600:.2f} h]'
        bar = ("\033[F\033[K " + progress_str + "\n"
               + elapsed_time_str
               + counter_str
               + percent_str
               + eta_str)
        print(bar, end=print_end)
        yield item


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
    try:
        for _ in ft_progress(range(max_iter)):
            gradient = gradient_(x, y, theta)
            if gradient is None:
                return None
            elif all(val == [0.] for val in gradient):
                break
            theta = theta - alpha * gradient
        print()
        return theta
    except Exception:
        return None


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
    theta2 = fit_(x, y, theta, alpha=0.0005, max_iter=42_000)
    print(theta2)
    # Output:
    # array([[41.99..],[0.97..], [0.77..], [-1.20..]])

    # Example 1:
    y_hat = predict_(x, theta2)
    print(y_hat)
    # Output:
    # array([[19.5992..], [-2.8003..], [-25.1999..], [-47.5996..]])
