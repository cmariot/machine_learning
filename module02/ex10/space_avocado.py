from matplotlib import pyplot as plt
import pandas
import numpy
from my_linear_regression import MyLR as MyLR
import yaml


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

    proportion_index = int(x.shape[0] * proportion)

    x_train = shuffled_x[:proportion_index]
    x_test = shuffled_x[proportion_index:]
    y_train = shuffled_y[:proportion_index]
    y_test = shuffled_y[proportion_index:]

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
    res = numpy.empty((x.shape[0], x.shape[1] * power))
    for i in range(x.shape[1]):
        for j in range(power):
            res[:, i * power + j] = x[:, i] ** (j + 1)
    return res


def print_model_info(model):
    print(model["name"].upper())
    print("price = θ0 + ", end="")
    (degree, degree2, degree3) = model["degres"]
    for i in range(1, degree + 1):
        print("θ" + str(i) + " * weight^" + str(i), end="")
        if i < degree:
            print(" + ", end="")
    print(" + ", end="")
    for i in range(1, degree2 + 1):
        print("θ" + str(i + degree) + " * prod_distance^" + str(i),
              end="")
        if i < degree2:
            print(" + ", end="")
    print(" + ", end="")
    for i in range(1, degree3 + 1):
        print("θ" + str(i + degree + degree2) +
              " * time_delivery^" + str(i), end="")
        if i < degree3:
            print(" + ", end="")
    print()


class Normalizer:

    def __init__(self, features_matrix) -> None:
        if not isinstance(features_matrix, numpy.ndarray):
            return None
        if features_matrix.size == 0 or features_matrix.shape[1] < 1:
            return None
        self.denormalized_features = features_matrix.copy()
        self.normalized_features = numpy.empty(features_matrix.shape)
        nb_features = features_matrix.shape[1]
        for feature_index in range(nb_features):
            self.normalized_features[:, feature_index] = \
                self.normalize(features_matrix[:, feature_index])
            if self.normalized_features[:, feature_index] is None:
                return None

    def normalize(self, x):
        if not isinstance(x, numpy.ndarray):
            return None
        if x.size == 0:
            return None
        min = numpy.min(x)
        max = numpy.max(x)
        return (x - min) / (max - min)


if __name__ == "__main__":

    # Get the dataset in the file space_avocado.csv
    dataset = get_dataset()

    # Split the dataset into a training and a test set
    splitted = data_spliter(dataset, 0.5)
    if splitted is None:
        exit(1)
    else:
        (x_train, x_test, y_train, y_test) = splitted

    linear_regression = MyLR(alpha=10e-2, max_iter=15_000)

    # Normalize the features and the target of the training set
    train_normalizer = Normalizer(x_train)
    x_train_normalized = train_normalizer.normalized_features
    y_train_normalized = linear_regression.normalize(y_train)

    test_normalizer = Normalizer(x_test)
    x_test_normalized = test_normalizer.normalized_features
    y_test_normalized = linear_regression.normalize(y_test)

    # Add polynomial features to the training set
    x_train_polynomial = add_polynomial_features(x_train_normalized, 4)
    x_test_polynomial = add_polynomial_features(x_test_normalized, 4)

    models = []
    max_degree = range(1, 4)
    for degree in max_degree:
        for degree2 in max_degree:
            for degree3 in max_degree:

                model = {}

                model["name"] = "w" + str(degree) + \
                                "d" + str(degree2) + \
                                "t" + str(degree3)
                model["degres"] = (degree, degree2, degree3)
                model["theta_shape"] = (degree + degree2 + degree3 + 1, 1)

                print_model_info(model)

                # ##################################### #
                # Train the model with the training set #
                # ##################################### #

                linear_regression.thetas = numpy.zeros(model["theta_shape"])

                weights = x_train_polynomial[:, 0:degree]
                prod = x_train_polynomial[:, 4:4 + degree2]
                time = x_train_polynomial[:, 8:8 + degree3]
                x_train_polynomial_model = numpy.concatenate(
                    (weights, prod, time), axis=1)

                linear_regression.fit_(
                    x_train_polynomial_model, y_train_normalized)

                model["theta"] = linear_regression.thetas

                # ############################# #
                # Evaluate the model prediction #
                # ############################# #

                weights = x_test_polynomial[:, :degree]
                prod = x_test_polynomial[:, 4:4 + degree2]
                time = x_test_polynomial[:, 8:8 + degree3]
                x_test_polynomial_model = numpy.concatenate(
                    (weights, prod, time), axis=1)

                y_hat_test = linear_regression.predict_(
                    x_test_polynomial_model)

                denormalized_y_hat_test = linear_regression.denormalize(
                    y_test, y_hat_test)

                model["cost"] = linear_regression.loss_(
                    y_test_normalized, y_hat_test)

                print(model)
                print()
                models.append(model)

                # ############################# #
                # Plot the results on 3 graphs  #
                # ############################# #

                fig, ax = plt.subplots(1, 3, figsize=(15, 7.5))

                ax[0].scatter(x_test[:, 0], y_test, color="blue")
                ax[0].scatter(x_test[:, 0], denormalized_y_hat_test,
                              color="red")
                ax[0].set_title("Weight")

                ax[1].scatter(x_test[:, 1], y_test, color="blue")
                ax[1].scatter(x_test[:, 1], denormalized_y_hat_test,
                              color="red")
                ax[1].set_title("Product distance")

                ax[2].scatter(x_test[:, 2], y_test, color="blue")
                ax[2].scatter(x_test[:, 2], denormalized_y_hat_test,
                              color="red")
                ax[2].set_title("Time delivery")

                plt.suptitle(model["name"] + " MSE : " + str(model["cost"]))
                plt.show()

    best_model = min(models, key=lambda x: x["cost"])

    # Save the models in the file models.yml
    with open("models.yml", "w") as file:
        file.write(yaml.dump(models))

    print("Best model is:")
    print(best_model)

    # ########################### #
    # Plot the cost of each model #
    # ########################### #

    plt.scatter(range(len(models)), [model["cost"] for model in models])
    plt.title("Cost of each model")
    plt.show()
