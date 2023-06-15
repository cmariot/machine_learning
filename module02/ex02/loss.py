import numpy as np


def loss_(y, y_hat):
    for arr in [y, y_hat]:
        if not isinstance(arr, np.ndarray):
            return None
    m = y.shape[0]
    if m == 0:
        return None
    if y.shape != (m, 1) or y_hat.shape != (m, 1):
        return None
    J_elem = np.square(y_hat - y)
    J_value = np.mean(J_elem) / 2
    return J_value


if __name__ == "__main__":

    X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((-1, 1))
    Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))

    # Example 1:
    print(loss_(X, Y))
    # Output:
    # 2.142857142857143
    # Example 2:
    print(loss_(X, X))
    # Output:
    # 0.0
