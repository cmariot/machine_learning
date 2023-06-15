import numpy as np


class MyLinearRegression():
    """
        Description:
            My personnal linear regression class to fit like a boss.
    """

    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        if (not isinstance(thetas, np.ndarray) or thetas.shape != (2, 1)):
            return None
        self.thetas = thetas
        if not isinstance(alpha, float) or alpha < 0.0:
            return None
        self.alpha = alpha
        if not isinstance(max_iter, int) or max_iter < 0:
            return None
        self.max_iter = max_iter

    def fit_(self, x, y):
        for arr in [x, y]:
            if not isinstance(arr, np.ndarray):
                return None
            if arr.size == 0:
                return None
        m = x.shape[0]
        if x.shape != (m, 1) or y.shape != (m, 1):
            return None
        Xprime = np.c_[np.ones((m, 1)), x]
        XprimeT = Xprime.T
        gradient = np.zeros((2, 1))
        for _ in range(self.max_iter):
            gradient = np.matmul((XprimeT), (Xprime.dot(self.thetas) - y)) / m
            if gradient[0] == 0. and gradient[1] == 0.:
                break
            self.thetas = self.thetas - self.alpha * gradient
        return self.thetas

    def predict_(self, x):
        """
        Computes the vector of prediction y_hat from two non-empty numpy.array.
        """
        if not isinstance(x, np.ndarray):
            return None
        m = x.shape[0]
        if m == 0 or x.shape != (m, 1):
            return None
        X = np.c_[np.ones(m), x]
        return X.dot(self.thetas)

    def loss_elem_(self, y, y_hat):
        for arr in [y, y_hat]:
            if not isinstance(arr, np.ndarray):
                return None
        m = y.shape[0]
        if m == 0:
            return None
        if y.shape != (m, 1) or y_hat.shape != (m, 1):
            return None
        return np.square(y_hat - y)

    def loss_(self, y, y_hat):
        J_elem = self.loss_elem_(y, y_hat)
        if J_elem is None:
            return None
        J_value = np.mean(J_elem) / 2
        return J_value


if __name__ == "__main__":

    x = np.array(
        [[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
    y = np.array(
        [[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])

    print("X :\n", x)
    print("Y :\n", y)

    lr1 = MyLinearRegression(np.array([[2], [0.7]]))
    print("Class Initialization : Thetas = [[2.0], [0.7]]"
          + " -> y_hat(x) = 2 + 0.7 * x")

    # Example 0.0:
    y_hat = lr1.predict_(x)
    print("y_hat with initial thetas :\n", y_hat)
    # Output:
    # array([[10.74695094],
    #        [17.05055804],
    #        [24.08691674],
    #        [36.24020866],
    #        [42.25621131]])

    # Example 0.1:
    loss_elem = lr1.loss_elem_(y, y_hat)
    # Output:
    # array([[710.45867381],
    #        [364.68645485],
    #        [469.96221651],
    #        [108.97553412],
    #        [299.37111101]])

    # Example 0.2:
    loss = lr1.loss_(y, y_hat)
    print("This prediction has a loss of :", loss)
    # Output:
    # 195.34539903032385

    # Example 1.0:
    lr2 = MyLinearRegression(np.array([[1], [1]]), 5e-8, 1500000)

    print("\nTraining the model ... Please wait.\n")

    lr2.fit_(x, y)

    print("After Fit : Thetas = [[{}], [{}]]".format(
        lr2.thetas[0],
        lr2.thetas[1]
        ))
    # Output:
    # array([[1.40709365],
    #        [1.1150909 ]])

    # Example 1.1:
    y_hat = lr2.predict_(x)
    print("Updated y_hat nearest to y :\n", y_hat)
    # Output:
    # array([[15.3408728 ],
    #       [25.38243697],
    #       [36.59126492],
    #       [55.95130097],
    #       [65.53471499]])

    # Example 1.2:
    loss_elem = lr2.loss_elem_(y, y_hat)
    # Output:
    # array([[486.66604863],
    #        [115.88278416],
    #        [ 84.16711596],
    #        [ 85.96919719],
    #        [ 35.71448348]])

    # Example 1.3:
    loss = lr2.loss_(y, y_hat)
    print("Updated loss, nearest to 0 :", loss)
    # Output:
    # 80.83996294128525
