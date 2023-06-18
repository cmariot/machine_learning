import numpy as np


class MyLR:

    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        if not isinstance(alpha, float) or alpha < 0.0:
            print("Error 2")
            return None
        elif not isinstance(max_iter, int) or max_iter < 0:
            print("Error 3")
            return None
        self.thetas = np.array(thetas)
        self.alpha = alpha
        self.max_iter = max_iter
        print("OK")

    def predict_(self, x):
        if not isinstance(x, np.ndarray):
            return None
        elif x.size == 0:
            return None
        m = x.shape[0]
        n = x.shape[1]
        if self.thetas.shape != (n + 1, 1):
            return None
        X_prime = np.concatenate((np.ones((m, 1)), x), axis=1)
        y_hat = np.dot(X_prime, self.thetas)
        return y_hat

    def loss_elem_(self, y, y_hat):
        for arr in [y, y_hat]:
            if not isinstance(arr, np.ndarray):
                return None
        m = y.shape[0]
        if m == 0:
            return None
        if y.shape != (m, 1) or y_hat.shape != (m, 1):
            return None
        J_elem = np.square(y_hat - y)
        return J_elem

    def loss_(self, y, y_hat):
        J_elem = self.loss_elem_(y, y_hat)
        if J_elem is None:
            return None
        J_value = np.mean(J_elem) / 2
        return J_value

    def gradient_(self, x, y):
        for array in [x, y]:
            if not isinstance(array, np.ndarray):
                return None
        m, n = x.shape
        if m == 0 or n == 0:
            return None
        elif y.shape != (m, 1):
            return None
        elif self.thetas.shape != (n + 1, 1):
            return None
        X_prime = np.c_[np.ones(m), x]
        return (1 / m) * (X_prime.T.dot(X_prime.dot(self.thetas) - y))

    def fit_(self, x, y):
        for arr in [x, y]:
            if not isinstance(arr, np.ndarray):
                return None
        m, n = x.shape
        if m == 0 or n == 0:
            return None
        if y.shape != (m, 1):
            return None
        elif self.thetas.shape != ((n + 1), 1):
            return None
        for _ in range(self.max_iter):
            gradient = self.gradient_(x, y)
            if gradient is None:
                return None
            if all(__ == 0. for __ in gradient):
                break
            self.thetas = self.thetas - self.alpha * gradient
        return self.thetas


if __name__ == "__main__":

    X = np.array([[1., 1., 2., 3.],
                  [5., 8., 13., 21.],
                  [34., 55., 89., 144.]])

    Y = np.array([[23.],
                  [48.],
                  [218.]])

    mylr = MyLR([[1.],
                 [1.],
                 [1.],
                 [1.],
                 [1]])

    # Example 0:
    y_hat = mylr.predict_(X)
    print(y_hat)
    # Output:
    # array([[8.], [48.], [323.]])

    # Example 1:
    mylr.loss_elem_(Y, y_hat)
    # Output:
    # array([[225.], [0.], [11025.]])

    # Example 2:
    loss = mylr.loss_(Y, y_hat)
    print(loss)
    # Output:
    # 1875.0

    # Example 3:
    mylr.alpha = 1.6e-4
    mylr.max_iter = 200000
    mylr.fit_(X, Y)
    print(mylr.thetas)
    # Output:
    # array([[18.188..], [2.767..], [-0.374..], [1.392..], [0.017..]])

    # Example 4:
    y_hat = mylr.predict_(X)
    print(y_hat)
    # Output:
    # array([[23.417..], [47.489..], [218.065...]])

    # Example 5:
    mylr.loss_elem_(Y, y_hat)
    # Output:
    # array([[0.174..], [0.260..], [0.004..]])

    # Example 6:
    loss = mylr.loss_(Y, y_hat)
    print(loss)
    # Output:
    # 0.0732..
