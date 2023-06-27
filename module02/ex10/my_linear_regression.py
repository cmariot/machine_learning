import numpy
import shutil
import time


class MyLR:

    def __init__(self, thetas, alpha=0.0001, max_iter=500000):
        if not isinstance(alpha, float) or alpha < 0.0:
            return None
        elif not isinstance(max_iter, int) or max_iter < 0:
            return None
        self.thetas = numpy.array(thetas)
        self.alpha = alpha
        self.max_iter = max_iter

    def predict_(self, x):
        if not isinstance(x, numpy.ndarray):
            return None
        elif x.size == 0:
            return None
        m = x.shape[0]
        n = x.shape[1]
        if self.thetas.shape != (n + 1, 1):
            return None
        X_prime = numpy.concatenate((numpy.ones((m, 1)), x), axis=1)
        y_hat = numpy.dot(X_prime, self.thetas)
        return y_hat

    def loss_elem_(self, y, y_hat):
        for arr in [y, y_hat]:
            if not isinstance(arr, numpy.ndarray):
                return None
        m = y.shape[0]
        if m == 0:
            return None
        if y.shape != (m, 1) or y_hat.shape != (m, 1):
            return None
        J_elem = numpy.square(y_hat - y)
        return J_elem

    def loss_(self, y, y_hat):
        J_elem = self.loss_elem_(y, y_hat)
        if J_elem is None:
            return None
        J_value = numpy.mean(J_elem) / 2
        return J_value

    def gradient_(self, x, y):
        for array in [x, y]:
            if not isinstance(array, numpy.ndarray):
                return None
        m, n = x.shape
        if m == 0 or n == 0:
            return None
        elif y.shape != (m, 1):
            return None
        elif self.thetas.shape != (n + 1, 1):
            return None
        X_prime = numpy.c_[numpy.ones(m), x]
        return (1 / m) * (X_prime.T.dot(X_prime.dot(self.thetas) - y))

    def ft_progress(self, iterable,
                    length=shutil.get_terminal_size().columns - 2,
                    fill='█',
                    empty='░',
                    print_end='\r'):
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

    def fit_(self, x, y, convergence=10e-10):
        for arr in [x, y]:
            if not isinstance(arr, numpy.ndarray):
                return None
        m, n = x.shape
        if m == 0 or n == 0:
            return None
        if y.shape != (m, 1):
            return None
        elif self.thetas.shape != ((n + 1), 1):
            return None
        print()
        for _ in self.ft_progress(range(self.max_iter)):
            gradient = self.gradient_(x, y)
            if gradient is None:
                return None
            if all(abs(__) <= convergence for __ in gradient):
                break
            self.thetas = self.thetas - self.alpha * gradient
        print()
        return self.thetas

    def mse_elem(self, y, y_hat) -> numpy.ndarray:
        return (y_hat - y) ** 2

    def mse_(self, y, y_hat) -> float:
        if any(not isinstance(_, numpy.ndarray) for _ in [y, y_hat]):
            return None
        m = y.shape[0]
        if m == 0 or y.shape != (m, 1) or y_hat.shape != (m, 1):
            return None
        J_elem = self.mse_elem(y, y_hat)
        return J_elem.mean()

    def minmax(self, x):
        if not isinstance(x, numpy.ndarray):
            return None
        if x.size == 0:
            return None
        return (x - numpy.min(x)) / (numpy.max(x) - numpy.min(x))

    def normalize(self, x):
        if not isinstance(x, numpy.ndarray):
            return None
        if x.size == 0:
            return None
        self.mean = numpy.mean(x)
        self.std = numpy.std(x)
        return (x - self.mean) / self.std

    def denormalize(self, x):
        if not isinstance(x, numpy.ndarray):
            return None
        if x.size == 0:
            return None
        return (x + self.mean) * self.std
