import shutil
import time
import pandas
import numpy
import matplotlib.pyplot as plt


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

    def fit_(self, x, y, convergence=10e-5):
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
        for _ in self.ft_progress(range(self.max_iter)):
            gradient = self.gradient_(x, y)
            if gradient is None:
                return None
            if all(abs(__) <= convergence for __ in gradient):
                break
            self.thetas = self.thetas - self.alpha * gradient
            print("gradient: ", gradient)
        print()
        print("AT THE END OF FIT : thetas: ", self.thetas)
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


def get_dataset():
    try:
        dataset = pandas.read_csv("../ressources/space_avocado.csv")
        dataset.drop(["Unnamed: 0"], axis=1, inplace=True)
        print(dataset)
        print(dataset.describe())
        return dataset
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

    if not isinstance(dataset, pandas.DataFrame):
        return None
    elif dataset.shape[0] == 0 or dataset.shape[1] != 4:
        return None
    elif not all(_ in dataset.columns
                 for _ in ["weight", "prod_distance",
                           "time_delivery", "target"]):
        return None

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


def add_polynomial_features(x, power):
    """
    Add polynomial features to vector x by raising its values up to
    the power given in argument.
    Args:
        x: has to be an numpy.array, a vector of dimension m * 1.
        power: has to be an int, the power up to which the components
               of vector x are going to be raised.
    Return:
        The matrix of polynomial features as a numpy.array, of dimension m * n,
        containing the polynomial feature values for all training examples.
        None if x is an empty numpy.array.
        None if x or power is not of expected type.
    Raises:
        This function should not raise any exception.
    """

    if not isinstance(x, numpy.ndarray) or not isinstance(power, int):
        return None
    elif x.size == 0 or power < 0:
        return None
    if power == 0:
        return numpy.ones((x.size, 1))
    res = numpy.ones((x.size, power))
    for i in range(1, power + 1):
        for j in range(x.size):
            res[j][i - 1] = x[j] ** i
    return res


if __name__ == "__main__":

    # Get the dataset in the file space_avocado.csv
    dataset = get_dataset()

    # Split the dataset into a training and a test set
    splitted = data_spliter(dataset, 0.1)
    if splitted is None:
        exit(1)
    else:
        (x_train, x_test, y_train, y_test) = splitted

    features = dataset.columns.values[:-1]
    nb_features = len(features)
    max_degree = 4

    # Use your polynomial_features method on your training set.
    # For each feature, create a polynomial of 4 degrees.

    linear_regression = MyLR(numpy.array((2, 1)), 5e-4, 1_000_000)
    x_train_poly = []
    for feature in range(nb_features):
        polynomial_feature = add_polynomial_features(
            x_train[:, feature], max_degree)
        # Normalize the training data
        for degree in range(max_degree):
            polynomial_feature[:, degree] = \
                linear_regression.minmax(polynomial_feature[:, degree])
        x_test[:, feature] = linear_regression.minmax(x_test[:, feature])
        x_train_poly.append(polynomial_feature)

    # Consider several Linear Regression models with polynomial hypothesis
    # with a maximum degree of 4.
    for degree in range(max_degree):
        print("Degree : {}".format(degree + 1))
        # Train your models on the training set.
        x = numpy.array([x_train_poly[0][:, degree],
                         x_train_poly[1][:, degree],
                         x_train_poly[2][:, degree]]).T
        y = y_train
        if degree == 0:
            thetas = numpy.array([[516625.14931463],
                                  [280804.06613411],
                                  [-26029.32304272],
                                  [3264.76650934]])
#         elif degree == 1:
#            [[ 5.54616088e+05]
#  [ 2.64783155e+05]
#  [-6.45475077e+01]
#  [-1.12745948e+04]]

        else:
            thetas = numpy.zeros((x.shape[1] + 1, 1))
        linear_regression.thetas = thetas
        y_hat = linear_regression.predict_(x)
        mse = linear_regression.mse_(y, y_hat)
        print("MSE : {}".format(mse))
        linear_regression.fit_(x, y)

        x_test = numpy.array([x_test[:, 0], x_test[:, 1], x_test[:, 2]]).T
        y_hat = linear_regression.predict_(x_test)
        mse = linear_regression.mse_(y_test, y_hat)
        print("MSE : {}".format(mse))

        fig, axs = plt.subplots(1, 3)
        y_hat_test = linear_regression.predict_(x_test)
        for feature in range(nb_features):
            axs[feature].scatter(x_test[:, feature], y_test)
            axs[feature].scatter(x_test[:, feature], y_hat_test,
                                 color='red', marker='.')
            axs[feature].set_title(features[feature])
        plt.show()


#    print("x_train shape : {}".format(x_train.shape))
#    print("y_train shape : {}".format(y_train.shape))
#    print("x_test shape : {}".format(x_test.shape))
#    print("y_test shape : {}".format(y_test.shape))
#
#    # Create a matrix, called polynomial_train_x to store the x training set,
#    # each index coreespond to a polynomial degree
#    # polynomial_train_x = []
#    # for feature in range(nb_features):
#    #     polynomial_x = add_polynomial_features(x_train[feature], 4)
#    #     if polynomial_x is None:
#    #         print("Error: can't add polynomial feature.")
#    #         exit(1)
#    #     polynomial_train_x.append(polynomial_x)
#
#    # print(len(polynomial_train_x))
#    # print(polynomial_train_x[:])
#
#    #for power in range(max_degree):
#
#    #    training_x = []
#
#    #    for feature in range(nb_features):
#    #        polynomial_x = polynomial_train_x[feature]
#    #        current_degree = polynomial_x[:, power]
#    #        training_x.append(current_degree)
#
#    #    new_array = numpy.array(training_x)
#    #    print("Shape =", new_array.shape)
#
#    y_hat_train = linear_regression.predict_(x_train)
#    print("MSE : {}\n".format(
#        linear_regression.mse_(y_train, y_hat_train)))
#
#    linear_regression.fit_(x_train, y_train)
#
#    print("theta : {}".format(linear_regression.thetas))
#
#    y_hat_train = linear_regression.predict_(x_train)
#    print("MSE : {}\n".format(
#        linear_regression.mse_(y_train, y_hat_train)))
#

