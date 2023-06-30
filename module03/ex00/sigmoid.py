import numpy as np


def sigmoid_(x):
    """
    Compute the sigmoid of a vector.
    """
    try:
        if not isinstance(x, np.ndarray):
            print("Error: x is not a np.array.")
            return None
        m = x.shape[0]
        if m == 0 or x.shape != (m, 1):
            print("Error: x is not a np.array of shape (m, 1).")
            return None
        return 1. / (1. + np.exp(-x))
    except Exception as e:
        print(e)
        return None


if __name__ == "__main__":

    x = np.array([[-4]])
    print(sigmoid_(x))

    x = np.array([[2]])
    print(sigmoid_(x))

    x = np.array([[-4], [2], [0]])
    print(sigmoid_(x))
