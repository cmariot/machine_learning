import shutil
import time
import numpy as np
import sklearn.linear_model as sklm


class MyRidge:

    def checkargs_init_(func):
        """
        Check the arguments of __init__ with a decorator.
        """
        def wrapper(self, theta, alpha=0.001, max_iter=1000, lambda_=0.5):
            try:
                if not isinstance(theta, np.ndarray):
                    raise TypeError("theta must be a numpy.ndarray")
                elif theta.shape[0] == 0 or theta.shape[1] != 1:
                    raise ValueError("theta must be a column vector")
                if not isinstance(alpha, (int, float)):
                    raise TypeError("alpha must be a float")
                elif alpha < 0:
                    raise ValueError("alpha must be positive")
                if not isinstance(max_iter, int):
                    raise TypeError("max_iter must be an integer")
                elif max_iter < 0:
                    raise ValueError("max_iter must be positive")
                if not isinstance(lambda_, (int, float)):
                    raise TypeError("lambda_ must be a float")
                elif lambda_ < 0:
                    raise ValueError("lambda_ must be positive")
                return func(self, theta, alpha, max_iter, lambda_)

            except Exception as e:
                print("MyRidge initialization error :", e)
                return None
        return wrapper

    @checkargs_init_
    def __init__(self, theta, alpha, max_iter, lambda_):
        try:
            self.theta = theta
            self.alpha = alpha
            self.max_iter = max_iter
            self.lambda_ = lambda_
            self.losses = []
        except Exception:
            return None

    def get_params_(self):
        """
        Returns a dictionary containing all parameters of the model.
        """
        try:
            return self.__dict__
        except Exception:
            return None

    def checkargs_set_params_(func):
        """
        Check the arguments of set_params_ with a decorator.
        """
        def wrapper(self, dict_):
            try:
                if not isinstance(dict_, dict):
                    raise TypeError("dict_ must be a dictionary")
                return func(self, dict_)
            except Exception as e:
                print("MyRidge set_params error :", e)
                return None
        return wrapper

    @checkargs_set_params_
    def set_params_(self, dict):
        """
        Set the parameters of the model.
        """
        try:
            for key, value in dict.items():
                if key == "theta":
                    self.theta = value
                elif key == "alpha":
                    self.alpha = value
                elif key == "max_iter":
                    self.max_iter = value
                elif key == "lambda_":
                    self.lambda_ = value
                elif key == "losses":
                    self.losses = value
            return self
        except Exception:
            return None

    def l2_(self):
        """
        Computes the L2 regularization
        """
        try:
            theta_prime = self.theta.copy()
            theta_prime[0, 0] = 0
            l2 = np.dot(theta_prime.T, theta_prime)
            return np.sum(l2)
        except Exception:
            return None

    def checkargs_loss_(func):
        def wrapper(self, y, y_hat):
            try:
                if not isinstance(y, np.ndarray) \
                        or not isinstance(y_hat, np.ndarray):
                    return None
                m = y.shape[0]
                n = self.theta.shape[0]
                if m == 0 or n == 0:
                    return None
                if y.shape != (m, 1) \
                    or y_hat.shape != (m, 1) \
                        or self.theta.shape != (n, 1):
                    return None
                return func(self, y, y_hat)
            except Exception:
                return None
        return wrapper

    @checkargs_loss_
    def loss_(self, y, y_hat):
        """
        Computes the regularized loss of a linear regression model
        from two non-empty numpy.array, without any for loop.
        Args:
            y: has to be an numpy.ndarray, a vector of shape m * 1.
            y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        Returns:
            The regularized loss as a float.
            None if y, y_hat, or theta are empty numpy.ndarray.
            None if y and y_hat do not share the same shapes.
        Raises:
            This function should not raise any Exception.
        """

        try:
            const = 1 / (2 * y.shape[0])
            loss_elem = self.loss_elem_(y, y_hat)
            if loss_elem is None:
                return None
            l2 = self.l2_()
            if l2 is None:
                return None
            regularization = self.lambda_ * l2
            return float(const * np.sum(loss_elem + regularization))
        except Exception:
            return None

    @checkargs_loss_
    def loss_elem_(self, y, y_hat):
        try:
            diff = y_hat - y
            loss_elem = (diff).T.dot(diff)
            return loss_elem
        except Exception:
            return None

    def checkargs_predict_(func):
        def wrapper(self, x):
            try:
                if not isinstance(x, np.ndarray):
                    return None
                n = x.shape[1]
                if n == 0:
                    return None
                if x.shape[0] == 0:
                    return None
                elif self.theta.shape != ((n + 1), 1):
                    return None
                return func(self, x)
            except Exception:
                return None
        return wrapper

    @checkargs_predict_
    def predict_(self, x):
        try:
            m = x.shape[0]
            x_prime = np.hstack((np.ones((m, 1)), x))
            y_hat = x_prime.dot(self.theta)
            return y_hat
        except Exception:
            return None

    def checkargs_gradient_(func):
        def wrapper(self, x, y):
            try:
                if not isinstance(y, np.ndarray) \
                        or not isinstance(x, np.ndarray):
                    return None
                m, n = x.shape
                if m == 0 or n == 0:
                    return None
                if y.shape != (m, 1) \
                    or x.shape != (m, n) \
                        or self.theta.shape != (n + 1, 1):
                    return None
                return func(self, x, y)
            except Exception:
                return None
        return wrapper

    @checkargs_gradient_
    def gradient_(self, x, y):
        """
        Computes the regularized linear gradient,
        without any for-loop.
        Args:
            y: has to be a numpy.ndarray, a vector of shape m * 1.
            x: has to be a numpy.ndarray, a matrix of dimesion m * n.
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
            theta_prime = self.theta.copy()
            theta_prime[0, 0] = 0.0
            m = x.shape[0]
            x_prime = np.hstack((np.ones((m, 1)), x))
            y_hat = x_prime.dot(self.theta)
            return (np.dot(x_prime.T, y_hat - y)
                    + (self.lambda_ * theta_prime)) / m
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

    def checkargs_fit_(func):
        def wrapper(self, x, y):
            try:
                if not isinstance(y, np.ndarray) \
                        or not isinstance(x, np.ndarray):
                    return None
                m, n = x.shape
                if m == 0 or n == 0:
                    return None
                if y.shape != (m, 1) \
                    or x.shape != (m, n) \
                        or self.theta.shape != (n + 1, 1):
                    return None
                return func(self, x, y)
            except Exception:
                return None
        return wrapper

    @checkargs_fit_
    def fit_(self, x, y):
        """
        Fits the model to the training dataset contained in x and y.
        """
        try:
            for _ in self.ft_progress(range(self.max_iter)):
                gradient = self.gradient_(x, y)
                if gradient is None:
                    return None
                self.theta = self.theta - (self.alpha * gradient)
                self.losses.append(self.loss_(y, self.predict_(x)))
            return self.theta
        except Exception:
            return None

    def r2score_elem(self, y, y_hat):
        try:
            m = y.shape[0]
            mean = y.mean()
            numerator, denominator = 0., 0.
            for i in range(m):
                numerator += (y_hat[i] - y[i]) ** 2
                denominator += (y[i] - mean) ** 2
            return numerator / denominator
        except Exception:
            return None

    def r2score_(self, y, y_hat):
        """
            Description:
                Calculate the R2score between y_hat and y.
            Args:
                y: has to be a numpy.array, a vector of dimension m * 1.
                y_hat: has to be a numpy.array, a vector of dimension m * 1.
            Returns:
                r2score: has to be a float.
                None if there is a matching dimension problem.
            Raises:
                This function should not raise any Exceptions.
        """
        try:
            if any(not isinstance(_, np.ndarray) for _ in [y, y_hat]):
                return None
            m, n = y.shape
            if m == 0 or y_hat.shape != (m, n) or n != 1:
                return None
            r2_score_elem = self.r2score_elem(y, y_hat)
            if r2_score_elem is None:
                return None
            return np.sum(1 - r2_score_elem)
        except Exception:
            return None


if __name__ == "__main__":

    ridge = sklm.Ridge(alpha=0.5)
    print(ridge.get_params())

    myridge = MyRidge(np.array([[1], [2]]))
    print(myridge.get_params_())

    y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    y_hat = np.array([3, 13, -11.5, 5, 11, 5, -20]).reshape((-1, 1))
    theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))

    dictionary: dict = {
        "theta": theta,
        "lambda_": 0.5
    }

    myridge.set_params_(dictionary)
    print(myridge.get_params_())

    # Example :
    print(myridge.loss_(y, y_hat))
    # Output:
    # 0.8503571428571429

    # Example :
    dictionary = {
        "theta": theta,
        "lambda_": 0.05
    }

    myridge.set_params_(dictionary)
    print(myridge.loss_(y, y_hat))
    # Output:
    # 0.5511071428571429

    # Example :
    dictionary = {
        "theta": theta,
        "lambda_": 0.9
    }

    myridge.set_params_(dictionary)
    print(myridge.loss_(y, y_hat))
    # Output:
    # 1.1163571428571428

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
    dictionary = {
        "theta": theta,
        "lambda_": 1
    }
    myridge.set_params_(dictionary)
    print(myridge.gradient_(x, y))
    # Output:
    # array([[ -60.99 ],
    #        [-195.64714286],
    #        [ 863.46571429],
    #        [-644.52142857]])

    # Example 2.1:
    dictionary = {
        "lambda_": 0.5
    }
    myridge.set_params_(dictionary)
    print(myridge.gradient_(x, y))
    # Output:
    # array([[ -60.99 ],
    #        [-195.86142857],
    #        [ 862.71571429],
    #        [-644.09285714]])

    # Example 3.1:
    dictionary = {
        "lambda_": 0
    }
    myridge.set_params_(dictionary)
    print(myridge.gradient_(x, y))
    # Output:
    # array([[ -60.99 ],
    #        [-196.07571429],
    #        [ 861.96571429],
    #        [-643.66428571]])
