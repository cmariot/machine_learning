import numpy as np


def loss_(y: np.ndarray, y_hat: np.ndarray):
    for arr in [y, y_hat]:
        if not isinstance(arr, np.ndarray):
            return None
    m = y.shape[0]
    if m == 0:
        return None
    if y.shape != (m, 1) or y_hat.shape != (m, 1):
        return None
    pred_sub_y = y_hat - y
    return (np.dot(pred_sub_y.T, pred_sub_y) / (2 * m))[0, 0]


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
