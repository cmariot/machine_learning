import numpy as np


def check_args(x, y, proportion):
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        return None
    m = x.shape[0]
    n = x.shape[1]
    if m == 0 or n == 0:
        return None
    elif y.shape != (m, 1):
        return None
    if not isinstance(proportion, (int, float)):
        return None
    elif proportion < 0.0 or proportion > 1.0:
        return None
    return True


def shuffle_data(x, y):
    shuffled_x = np.empty(x.shape, dtype=x.dtype)
    shuffled_y = np.empty(y.shape, dtype=y.dtype)
    m = x.shape[0]
    available_indexes = np.arange(m)
    for i in range(m):
        # Pick a random index in the available indexes and remove it
        index = np.random.choice(available_indexes)
        available_indexes = np.delete(available_indexes,
                                      np.where(available_indexes == index))
        shuffled_x[i] = x[index]
        shuffled_y[i] = y[index]
    return (shuffled_x, shuffled_y)


def print_data(data, name):
    print(name + " : ", end="")
    for i, val in enumerate(data):
        if i == len(data) - 1:
            print(val)
        else:
            print(val, end=", ")


def data_spliter(x, y, proportion):
    """
    Shuffles and splits the dataset (given by x and y) into a training
    and a test set, while respecting the given proportion of examples to
    be kept in the training set.
    Args:
        x: has to be an numpy.array, a matrix of dimension m * n.
        y: has to be an numpy.array, a vector of dimension m * 1.
        proportion: has to be a float, the proportion of the dataset that will
                    be assigned to the training set.
    Return:
        (x_train, x_test, y_train, y_test) as a tuple of numpy.array
        None if x or y is an empty numpy.array.
        None if x and y do not share compatible dimensions.
        None if x, y or proportion is not of expected type.
    Raises:
        This function should not raise any Exception.
    """

    if check_args(x, y, proportion) is None:
        return None

    shuffled_x, shuffled_y = shuffle_data(x, y)

    try:
        proportion_index = int(x.shape[0] * proportion)

        x_train = shuffled_x[:proportion_index]
        y_train = shuffled_y[:proportion_index]
        x_test = shuffled_x[proportion_index:]
        y_test = shuffled_y[proportion_index:]
        return (x_train, x_test, y_train, y_test)
    except Exception:
        return None


if __name__ == "__main__":

    # Create a matrix of dimension (10, 2) with random values
    x = np.random.randint(0, 100, (100, 2))

    # Create a vector of dimension (10, 1)
    y = np.random.randint(0, 100, (100, 1))

    print("x.shape : ", x.shape)
    print("y.shape : ", y.shape)

    print("Splitting the dataset into a training set and a test set ...")
    (x_train, x_test, y_train, y_test) = data_spliter(x, y, 0.2)

    # Print the training set
    print_data(x_train, "taining_x")
    print_data(y_train, "training_y")

    print("taining_x.shape : ", x_train.shape)
    print("taining_y.shape : ", y_train.shape)

    # Print the test set
    print_data(x_test, "test_x")
    print_data(y_test, "test_y")

    print("test_x.shape : ", x_test.shape)
    print("test_y.shape : ", y_test.shape)
