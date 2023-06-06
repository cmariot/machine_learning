import numpy as np
from typing import Union


class MyLinearRegression():
    """
        Description:
                My personnal linear regression class to fit like a boss.
    """

    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        # Args checking :
        if not isinstance(thetas, np.ndarray):
            return None
        if thetas.shape != (2, 1) and thetas.shape != (2,):
            return None
        if not isinstance(alpha, float):
            return None
        if not isinstance(max_iter, int):
            return None
        if alpha < 0:
            return None
        if max_iter < 0:
            return None

        self.thetas = thetas
        self.alpha = alpha
        self.max_iter = max_iter

    def fit_(self, x, y):
        for arr in [x, y, self.thetas]:
            if not isinstance(arr, np.ndarray):
                return None
            if arr.size == 0:
                return None

        # Check if x and y have compatible shapes
        m = x.shape[0]
        if x.shape != y.shape or x.shape[1] != 1 \
                or self.thetas.shape != (2, 1):
            return None

        # Reshape the x array to be a matrix of shape m * 2
        X = np.c_[np.ones((m, 1)), x]
        X = X.reshape((m, 2))

        # Gradient descent
        self.thetas = self.thetas.reshape((2, 1))
        for _ in range(self.max_iter):
            gradient = np.zeros((2, 1))
            gradient[0] = np.sum((X.dot(self.thetas) - y)) / m
            gradient[1] = np.sum((X.dot(self.thetas) - y) * x) / m
            self.thetas = self.thetas - self.alpha * gradient

        return self.thetas

    def predict_(self, x) -> Union[np.ndarray, None]:
        """
        Computes the vector of prediction y_hat from two non-empty numpy.array.
        """
        if not isinstance(x, np.ndarray):
            return None
        elif x.size == 0 or x.shape[1] != 1:
            return None
        X = np.c_[np.ones(x.shape[0]), x]
        y_hat = X.dot(self.thetas)

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
        return (y_hat - y) ** 2

    def loss_(self, y, y_hat):
        # Compute the loss by calling loss_elem_
        J_elem = self.loss_elem_(y, y_hat)
        if J_elem is None:
            return None
        # Compute the mean of J_elem
        J_value = np.mean(J_elem)
        return J_value / 2


if __name__ == "__main__":

    x = np.array(
        [[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
    y = np.array(
        [[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])

    lr1 = MyLinearRegression(np.array([[2], [0.7]]))

    print("THETAS :", lr1.thetas)

    # Example 0.0:
    y_hat = lr1.predict_(x)
    print("Y_HAT :", y_hat)
    # Output:
    # array([[10.74695094],
    #        [17.05055804],
    #        [24.08691674],
    #        [36.24020866],
    #        [42.25621131]])

    # Example 0.1:
    loss_elem = lr1.loss_elem_(y, y_hat)
    print("LOSS_ELEM :", loss_elem)
    # Output:
    # array([[710.45867381],
    #        [364.68645485],
    #        [469.96221651],
    #        [108.97553412],
    #        [299.37111101]])

    # Example 0.2:
    loss = lr1.loss_(y, y_hat)
    print("LOSS :", loss)
    # Output:
    # 195.34539903032385

    # Example 1.0:
    lr2 = MyLinearRegression(np.array([[1], [1]]), 5e-8, 1500000)
    lr2.fit_(x, y)
    print("PREDICTED THETAS :", lr2.thetas)
    # Output:
    # array([[1.40709365],
    #        [1.1150909 ]])

    # Example 1.1:
    y_hat = lr2.predict_(x)
    print("NEW Y_HAT :", y_hat)
    # Output:
    # array([[15.3408728 ],
    #       [25.38243697],
    #       [36.59126492],
    #       [55.95130097],
    #       [65.53471499]])

    # Example 1.2:
    loss_elem = lr2.loss_elem_(y, y_hat)
    print("NEW LOSS_ELEM :", loss_elem)
    # Output:
    # array([[486.66604863],
    #        [115.88278416],
    #        [ 84.16711596],
    #        [ 85.96919719],
    #        [ 35.71448348]])

    # Example 1.3:
    loss = lr2.loss_(y, y_hat)
    print("NEW LOSS :", loss)
    # Output:
    # 80.83996294128525
