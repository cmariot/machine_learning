import pandas
import numpy


def get_dataset():
    try:
        return pandas.read_csv("../ressources/space_avocado.csv")
    except Exception:
        print("Error: Can't find the dataset file")
        exit(1)


def check_args(x, y, proportion):
    if not isinstance(x, numpy.ndarray) or not isinstance(y, numpy.ndarray):
        return None
    m = x.shape[0]
    n = x.shape[1]
    if m == 0 or n == 0:
        return None
    elif y.shape != (m, 1):
        print("Error: y.shape != (m, 1)"
              + str(y.shape)
              + " != "
              + str((m, 1)))
        return None
    if not isinstance(proportion, float):
        return None
    elif proportion < 0.0 or proportion > 1.0:
        return None
    return True


def shuffle_data(x, y):
    shuffled_x = numpy.empty(x.shape, dtype=x.dtype)
    shuffled_y = numpy.empty(y.shape, dtype=y.dtype)
    m = x.shape[0]
    available_indexes = numpy.arange(m)
    for i in range(m):

        # Pick a random index in the available indexes and remove it
        index = numpy.random.choice(available_indexes)
        available_indexes = numpy.delete(available_indexes,
                                         numpy.where(available_indexes
                                                     == index))
        shuffled_x[i] = x[index]
        shuffled_y[i] = y[index]
    return (shuffled_x, shuffled_y)


def data_spliter(dataset, proportion):
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

    weight = dataset["weight"].values
    prod_distance = dataset["prod_distance"].values
    time_delivery = dataset["time_delivery"].values

    target = dataset["target"].values

    x = numpy.array([weight, prod_distance, time_delivery]).T
    y = numpy.array(target).reshape(-1, 1)

    if check_args(x, y, proportion) is None:
        return None

    shuffled_x, shuffled_y = shuffle_data(x, y)

    proportin_index = int(x.shape[0] * proportion)

    x_train = shuffled_x[:proportin_index]
    x_test = shuffled_x[proportin_index:]
    y_train = shuffled_y[:proportin_index]
    y_test = shuffled_y[proportin_index:]

    return (x_train, x_test, y_train, y_test)


if __name__ == "__main__":

    # Get the dataset in the file space_avocado.csv
    dataset = get_dataset()

    # Split the dataset into a training and a test set
    (x_train, x_test, y_train, y_test) = data_spliter(dataset, 0.8)

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
