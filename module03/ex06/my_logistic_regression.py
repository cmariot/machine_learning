import shutil
import time
import numpy as np
import sklearn.metrics as skm


class MyLogisticRegression():
    """
        Description:
            My personnal logistic regression to classify things.
    """

    def __init__(self, theta, alpha=0.001, max_iter=1000):

        if not isinstance(theta, np.ndarray):
            return None
        if not isinstance(alpha, (int, float)):
            return None
        if not isinstance(max_iter, int):
            return None
        if max_iter <= 0:
            return None
        if alpha <= 0:
            return None

        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = theta

    @staticmethod
    def sigmoid_(x):
        """
        Compute the sigmoid of a vector.
        Args:
            x: has to be a numpy.ndarray.
        Returns:
            The sigmoid value as a numpy.ndarray.
            None if an exception occurs.
        Raises:
            This function should not raise any Exception.
        """
        try:
            return 1 / (1 + np.exp(-x))
        except Exception:
            return None

    def predict_(self, x):
        """
        Computes the vector of prediction y_hat from two
        non-empty numpy.ndarray.
        Args:
        x: has to be an numpy.ndarray, a vector of dimension m * n.
        Returns:
        y_hat as a numpy.ndarray, a vector of dimension m * 1.
        None if x or theta are empty numpy.ndarray.
        None if x or theta dimensions are not appropriate.
        Raises:
        This function should not raise any Exception.
        """

        if not isinstance(x, np.ndarray):
            return None

        try:
            m, n = x.shape

            if m == 0 or n == 0:
                return None
            elif self.theta.shape != (n + 1, 1):
                return None

            X_prime = np.hstack((np.ones((m, 1)), x))
            y_hat = self.sigmoid_(X_prime.dot(self.theta))
            return y_hat

        except Exception:
            return None

    def loss_elem_(self, y, y_hat, eps=1e-15):
        try:
            if not all(isinstance(x, np.ndarray) for x in [y, y_hat]):
                return None

            m, n = y.shape
            if (m == 0 or n == 0):
                return None
            elif y_hat.shape != (m, n):
                return None

            y_hat[y_hat == 0] = eps
            y_hat[y_hat == 1] = 1 - eps

            dot1 = y.T.dot(np.log(y_hat))
            dot2 = (1 - y).T.dot(np.log(1 - y_hat))
            return (dot1 + dot2)

        except Exception:
            return None

    def loss_(self, y, y_hat, eps=1e-15):
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

        try:
            loss_elem = self.loss_elem_(y, y_hat, eps)
            if loss_elem is None:
                return None
            return (loss_elem / -y.shape[0]).sum()

        except Exception:
            return None

    def ft_progress(self, iterable,
                    length=shutil.get_terminal_size().columns - 2,
                    fill='█',
                    empty='░',
                    print_end='\r'):
        total = len(iterable)
        start = time.time()
        print()
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
            progress_str = str(fill * filled_length
                               + empty * (length - filled_length))
            counter_str = f' [{i:>{len(str(total))}}/{total}] '
            if elapsed_time < 60:
                et_str = f' [Elapsed-time {elapsed_time:.2f} s]'
            elif elapsed_time < 3600:
                et_str = f' [Elapsed-time {elapsed_time / 60:.2f} m]'
            else:
                et_str = f' [Elapsed-time {elapsed_time / 3600:.2f} h]'
            bar = ("\033[F\033[K " + progress_str + "\n"
                   + et_str
                   + counter_str
                   + percent_str
                   + eta_str)
            print(bar, end=print_end)
            yield item
        print()

    def gradient_(self, x, y):
        """
        Computes a gradient vector from three non-empty numpy.ndarray,
        without any for-loop.
        The three arrays must have compatible shapes.
        Args:
            x: has to be an numpy.ndarray, a matrix of shape m * n.
            y: has to be an numpy.ndarray, a vector of shape m * 1.
        Returns:
            The gradient as a numpy.ndarray, a vector of shape n * 1,
                containg the result of the formula for all j.
            None if x, y, or theta are empty numpy.ndarray.
            None if x, y and theta do not have compatible shapes.
        Raises:
            This function should not raise any Exception.
        """

        try:
            if not all(isinstance(arr, np.ndarray) for arr in [x, y]):
                return None

            m, n = x.shape

            if m == 0 or n == 0:
                return None
            elif y.shape != (m, 1) or self.theta.shape != ((n + 1), 1):
                return None

            X_prime = np.hstack((np.ones((m, 1)), x))
            y_hat = self.predict_(x)
            if y_hat is None:
                return None
            return (X_prime.T.dot(y_hat - y)) / m

        except Exception:
            return None

    def fit_(self, x, y):
        for arr in [x, y]:
            if not isinstance(arr, np.ndarray):
                return None
        m, n = x.shape
        if m == 0 or n == 0:
            return None
        if y.shape != (m, 1):
            return None
        elif self.theta.shape != ((n + 1), 1):
            return None
        for _ in self.ft_progress(range(self.max_iter)):
            gradient = self.gradient_(x, y)
            if gradient is None:
                return None
            elif all(__ == 0.0 for __ in gradient):
                break
            self.theta = self.theta - self.alpha * gradient
        return self.theta


if __name__ == "__main__":
    X = np.array([[1., 1., 2., 3.],
                  [5., 8., 13., 21.],
                  [3., 5., 9., 14.]])
    Y = np.array([[1], [0], [1]])
    theta = np.array([[2], [0.5], [7.1], [-4.3], [2.09]])

    mylr = MyLogisticRegression(theta)

    # Example 0:
    y_hat = mylr.predict_(X)
    print(y_hat)
    # Output:
    # array([[0.99930437],
    #        [1.        ],
    #        [1. ]])

    # Example 1:
    sklearn_loss = skm.log_loss(Y, y_hat, eps=1e-15, labels=[0, 1])
    loss = mylr.loss_(Y, y_hat)
    print(loss, "vs", sklearn_loss)
    # Output:
    # 11.513157421577004

    # Example 2:
    mylr.fit_(X, Y)
    print(mylr.theta)
    # Output:
    # array([[ 2.11826435]
    #        [ 0.10154334]
    #        [ 6.43942899]
    #        [-5.10817488]
    #        [ 0.6212541 ]])

    # Example 3:
    y_hat = mylr.predict_(X)
    print(y_hat)
    # Output:
    # array([[0.57606717]
    #       [0.68599807]
    #       [0.06562156]])

    # Example 4:
    sklearn_loss = skm.log_loss(Y, y_hat, eps=1e-15, labels=[0, 1])
    loss = mylr.loss_(Y, y_hat)
    print(loss, "vs", sklearn_loss)
    # Output:
    # 1.4779126923052268
