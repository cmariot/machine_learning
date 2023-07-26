import shutil
import time
import numpy as np
import sklearn.metrics as skm


class MyLogisticRegression:
    """
    Description: My personnal logistic regression to classify things.
    """

    # We consider l2 penality only. One may wants to implement other penalities
    supported_penalities = ['l2']

    # Check on type, data type, value ... if necessary
    def __init__(self, theta, alpha=0.001, max_iter=1000,
                 penality='l2', lambda_=1.0):
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = theta
        self.penality = penality
        self.lambda_ = lambda_ if penality in self.supported_penalities \
            else 0.0
        self.losses = []

    def checkargs_sigmoid_(func):
        def wrapper(self, x):
            try:
                if not isinstance(x, np.ndarray):
                    raise TypeError(
                        "x must be a numpy.ndarray")
                m = x.shape[0]
                if m == 0 or x.shape != (m, 1):
                    raise ValueError(
                        "x must be a numpy.ndarray of shape (m, 1)")
                return func(self, x)
            except Exception as e:
                print("MyLogisticRegression sigmoid_ error :", e)
                return None
        return wrapper

    @checkargs_sigmoid_
    def sigmoid_(self, x):
        try:
            return 1 / (1 + np.exp(-x))
        except Exception:
            return None

    def checkargs_predict_(func):
        def wrapper(self, x):
            try:
                if not isinstance(x, np.ndarray):
                    print("x is not a np.ndarray")
                    return None
                m, n = x.shape
                if m == 0 or n == 0:
                    print("m or n is 0")
                    return None
                elif self.theta.shape != ((n + 1), 1):
                    print("theta has a wrong shape")
                    return None
                return func(self, x)
            except Exception as e:
                print("CHECK:", e)
                return None
        return wrapper

    @checkargs_predict_
    def predict_(self, x):
        try:
            m = x.shape[0]
            x_prime = np.hstack((np.ones((m, 1)), x))
            y_hat = self.sigmoid_(x_prime.dot(self.theta))
            return y_hat
        except Exception as e:
            print("PREDICT:", e)
            return None

    def checkargs_l2_(func):
        def wrapper(self, theta):
            if not isinstance(theta, np.ndarray):
                return None
            elif theta.size == 0:
                return None
            return func(self, theta)
        return wrapper

    @checkargs_l2_
    def l2(self, theta):
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
        def wrapper(self, y, y_hat):
            try:
                if not isinstance(y, np.ndarray) \
                    or not isinstance(y_hat, np.ndarray) \
                        or not isinstance(self.theta, np.ndarray):
                    print("y, y_hat or theta is not a np.ndarray")
                    return None
                m = y.shape[0]
                n = self.theta.shape[0]
                if m == 0 or n == 0:
                    print("m or n is 0")
                    return None
                if y.shape != (m, 1) \
                    or y_hat.shape != (m, 1) \
                        or self.theta.shape != (n, 1):
                    print("y, y_hat or theta has a wrong shape")
                    return None
                return func(self, y, y_hat)
            except Exception as e:
                print(e)
                return None
        return wrapper

    def vec_log_loss_(self, y, y_hat, eps=1e-15):
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
    def loss_(self, y, y_hat):
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
            loss = self.vec_log_loss_(y, y_hat)
            if loss is None:
                print("loss is None")
                return None
            l2_ = self.l2(self.theta)
            if l2_ is None:
                print("l2_ is None")
                return None
            reg = (self.lambda_ / (2 * y.shape[0])) * l2_
            return loss + reg
        except Exception as e:
            print(e)
            return None

    def checkargs_gradient_(func):
        def wrapper(self, x, y):
            try:
                if not isinstance(y, np.ndarray) \
                        or not isinstance(x, np.ndarray):
                    print("y or x is not a np.ndarray")
                    return None
                m, n = x.shape
                if m == 0 or n == 0:
                    print("m or n is 0")
                    return None
                if y.shape != (m, 1) \
                    or x.shape != (m, n) \
                        or self.theta.shape != (n + 1, 1):
                    print("y, x or theta has a wrong shape")
                    return None
                return func(self, x, y)
            except Exception as e:
                print(e)
                return None
        return wrapper

    @checkargs_gradient_
    def gradient_(self, x, y):
        try:
            m, _ = x.shape
            X_prime = np.hstack((np.ones((m, 1)), x))
            y_hat = self.predict_(x)
            if y_hat is None:
                print("y_hat is None")
                return None
            theta_prime = self.theta.copy()
            theta_prime[0, 0] = 0.0
            return (X_prime.T.dot(y_hat - y) + (self.lambda_ * theta_prime)) \
                / m
        except Exception as e:
            print(e)
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
            except Exception as e:
                print(e)
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
                y_hat = self.predict_(x)
                self.losses.append(self.loss_(y, y_hat))
            return self.theta
        except Exception:
            return None

    def accuracy_score_(self, y, y_hat):
        """
        Compute the accuracy score.
        Accuracy tells you the percentage of predictions that are accurate
        (i.e. the correct class was predicted).
        Accuracy doesn't give information about either error type.
        Args:
            y: a numpy.ndarray for the correct labels
            y_hat:a numpy.ndarray for the predicted labels
        Returns:
            The accuracy score as a float.
            None on any error.
        Raises:
            This function should not raise any Exception.
        """

        try:
            if not isinstance(y, np.ndarray) \
                    or not isinstance(y_hat, np.ndarray):
                return None

            if y.shape != y_hat.shape:
                return None

            if y.size == 0:
                return None

            true = np.where(y == y_hat)[0].shape[0]
            return true / y.size

        except Exception:
            return None

    def precision_score_(self, y, y_hat, pos_label=1):
        """
        Compute the precision score.
        Precision tells you how much you can trust your
        model when it says that an object belongs to Class A.
        More precisely, it is the percentage of the objects
        assigned to Class A that really were A objects.
        You use precision when you want to control for False positives.
        Args:
            y: a numpy.ndarray for the correct labels
            y_hat: a numpy.ndarray for the predicted labels
            pos_label: str or int, the class on which to report
                    the precision_score (default=1)
        Return:
            The precision score as a float.
            None on any error.
        Raises:
            This function should not raise any Exception.
        """
        try:
            if not isinstance(y, np.ndarray) \
                    or not isinstance(y_hat, np.ndarray):
                return None

            if y.shape != y_hat.shape:
                return None

            if y.size == 0 or y_hat.size == 0:
                return None

            if not isinstance(pos_label, (int, str)):
                return None

            tp = np.sum(np.logical_and(y == pos_label, y == y_hat))
            fp = np.sum(np.logical_and(y != pos_label, y_hat == pos_label))
            return tp / (tp + fp)

        except Exception:
            return None

    def recall_score_(self, y, y_hat, pos_label=1):
        """
        Compute the recall score.
        Recall tells you how much you can trust that your
        model is able to recognize ALL Class A objects.
        It is the percentage of all A objects that were properly
        classified by the model as Class A.
        You use recall when you want to control for False negatives.
        Args:
            y:a numpy.ndarray for the correct labels
            y_hat:a numpy.ndarray for the predicted labels
            pos_label: str or int, the class on which to report
                    the precision_score (default=1)
        Return:
            The recall score as a float.
            None on any error.
        Raises:
            This function should not raise any Exception.
        """
        try:
            if not isinstance(y, np.ndarray) \
                    or not isinstance(y_hat, np.ndarray):
                return None

            if y.shape != y_hat.shape:
                return None

            if y.size == 0 or y_hat.size == 0:
                return None

            if not isinstance(pos_label, (int, str)):
                return None

            tp = np.sum(np.logical_and(y == pos_label, y == y_hat))
            fn = np.sum(np.logical_and(y == pos_label, y_hat != pos_label))
            return tp / (tp + fn)

        except Exception:
            return None

    def f1_score_(self, y, y_hat, pos_label=1):
        """
        Compute the f1 score.
        F1 score combines precision and recall in one single measure.
        You use the F1 score when want to control both
        False positives and False negatives.
        Args:
            y: a numpy.ndarray for the correct labels
            y_hat: a numpy.ndarray for the predicted labels
            pos_label: str or int, the class on which to report
                    the precision_score (default=1)
        Returns:
            The f1 score as a float.
            None on any error.
        Raises:
            This function should not raise any Exception.
        """
        try:
            if not isinstance(y, np.ndarray) \
                    or not isinstance(y_hat, np.ndarray):
                return None

            if y.shape != y_hat.shape:
                return None

            if y.size == 0 or y_hat.size == 0:
                return None

            if not isinstance(pos_label, (int, str)):
                return None

            precision = self.precision_score_(y, y_hat, pos_label)
            recall = self.recall_score_(y, y_hat, pos_label)
            return 2 * (precision * recall) / (precision + recall)

        except Exception as e:
            print(e)
            return None


if __name__ == "__main__":

    y = np.array([1, 1, 0, 0, 1, 1, 0]).reshape((-1, 1))
    y_hat = np.array([.9, .79, .12, .04, .89, .93, .01]).reshape((-1, 1))
    theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))

    # Example :
    mylr = MyLogisticRegression(theta=theta, lambda_=.5)
    print(mylr.loss_(y, y_hat))
    reg_term = (.5 / (2 * y.shape[0])) * mylr.l2(theta)
    print(skm.log_loss(y, y_hat, eps=1e-15, labels=[0, 1]) + reg_term)
    # Output:
    # 0.43377043716475955

    # Example :
    mylr = MyLogisticRegression(theta=theta, lambda_=.05)
    print(mylr.loss_(y, y_hat))
    reg_term = (.05 / (2 * y.shape[0])) * mylr.l2(theta)
    print(skm.log_loss(y, y_hat, eps=1e-15, labels=[0, 1]) + reg_term)
    # Output:
    # 0.13452043716475953

    # Example :

    mylr = MyLogisticRegression(theta=theta, lambda_=.9)
    print(mylr.loss_(y, y_hat))
    reg_term = (.9 / (2 * y.shape[0])) * mylr.l2(theta)
    print(skm.log_loss(y, y_hat, eps=1e-15, labels=[0, 1]) + reg_term)
    # Output:
    # 0.6997704371647596

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
    mylr.theta = theta
    mylr.lambda_ = 1
    print(mylr.gradient_(x, y))
    # Output:
    # array([[ -60.99 ],
    #        [-195.64714286],
    #        [ 863.46571429],
    #        [-644.52142857]])

    # Example 2.1:
    mylr.lambda_ = 0.5
    print(mylr.gradient_(x, y))
    # Output:
    # array([[ -60.99 ],
    #        [-195.86142857],
    #        [ 862.71571429],
    #        [-644.09285714]])

    # Example 3.1:
    mylr.lambda_ = 0
    print(mylr.gradient_(x, y))
    # Output:
    # array([[ -60.99 ],
    #        [-196.07571429],
    #        [ 861.96571429],
    #        [-643.66428571]])

    x = np.array([[0, 2, 3, 4],
                  [2, 4, 5, 5],
                  [1, 3, 2, 7]])

    y = np.array([[0], [1], [1]])

    theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])

    # Example 1.1:
    mylr = MyLogisticRegression(theta=theta, lambda_=1)
    print(mylr.gradient_(x, y))
    # Output:
    # array([[-0.55711039],
    #     [-1.40334809],
    #     [-1.91756886],
    #     [-2.56737958],
    #     [-3.03924017]])

    # Example 2.1:
    mylr = MyLogisticRegression(theta=theta, lambda_=0.5)
    print(mylr.gradient_(x, y))
    # Output:
    # array([[-0.55711039],
    #     [-1.15334809],
    #     [-1.96756886],
    #     [-2.33404624],
    #     [-3.15590684]])

    # Example 3.1:
    mylr = MyLogisticRegression(theta=theta, lambda_=0)
    print(mylr.gradient_(x, y))
    # Output:
    # array([[-0.55711039],
    #     [-0.90334809],
    #     [-2.01756886],
    #     [-2.10071291],
    #     [-3.27257351]])

    theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])

    # Example 1:
    model1 = MyLogisticRegression(theta, lambda_=5.0)
    print(model1.penality)
    # Output
    # ’l2’
    print(model1.lambda_)
    # Output
    # 5.0

    # Example 2:
    model2 = MyLogisticRegression(theta, penality=None)
    print(model2.penality)
    # Output
    # None
    print(model2.lambda_)
    # Output
    # 0.0

    # Example 3:
    model3 = MyLogisticRegression(theta, penality=None, lambda_=2.0)
    print(model3.penality)
    # Output
    # None
    print(model3.lambda_)
    # Output
    # 0.0
