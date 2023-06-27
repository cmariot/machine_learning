from matplotlib import pyplot as plt
import pandas
import numpy
from my_linear_regression import MyLR as MyLR
import json


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
    splitted = data_spliter(dataset, 0.5)
    if splitted is None:
        exit(1)
    else:
        (x_train, x_test, y_train, y_test) = splitted

    max_degree = 4
    features = dataset.columns.values[:-1]
    nb_features = len(features)

    linear_regression = MyLR(numpy.array((2, 1)), 0.05, 10_000)

    # Normalize the features
    for feature_index in range(nb_features):
        x_train[:, feature_index] = \
            linear_regression.normalize(x_train[:, feature_index])
        x_test[:, feature_index] = \
            linear_regression.normalize(x_test[:, feature_index])
    y_train = linear_regression.normalize(y_train)
    y_test = linear_regression.normalize(y_test)

    polynomial_weight = add_polynomial_features(x_train[:, 0], 4)
    polynomial_prod_distance = add_polynomial_features(x_train[:, 1], 4)
    polynomial_time_delivery = add_polynomial_features(x_train[:, 2], 4)

    test_polynolial_weight = add_polynomial_features(x_test[:, 0], 4)
    test_polynomial_prod_distance = add_polynomial_features(x_test[:, 1], 4)
    test_polynomial_time_delivery = add_polynomial_features(x_test[:, 2], 4)

    models = []

    for degree in range(1, 5):
        for degree2 in range(1, 5):
            for degree3 in range(1, 5):

                model = {}
                model["name"] = "w" + str(degree) + \
                                "d" + str(degree2) + \
                                "t" + str(degree3)
                model["degres"] = (degree, degree2, degree3)

                model_weight = polynomial_weight[:, :degree]
                model_prod_distance = polynomial_prod_distance[:, :degree2]
                model_time_delivery = polynomial_time_delivery[:, :degree3]

                x_train = numpy.concatenate(
                    (model_weight,
                     model_prod_distance,
                     model_time_delivery),
                    axis=1)

                theta_shape = (degree + degree2 + degree3 + 1, 1)
                linear_regression.thetas = numpy.ones(theta_shape)
                model["theta_shape"] = theta_shape

                linear_regression.fit_(x_train, y_train)
                predicted_theta = linear_regression.thetas
                model["theta"] = predicted_theta.tolist()

                # Create an array with the polynomial features from 0 to degree
                # for each feature for the test set
                test_weight = test_polynolial_weight[:, :degree]
                test_prod_distance = test_polynomial_prod_distance[:, :degree2]
                test_time_delivery = test_polynomial_time_delivery[:, :degree3]

                x_test = numpy.concatenate(
                    (test_weight,
                        test_prod_distance,
                        test_time_delivery),
                    axis=1)

                y_hat = linear_regression.predict_(x_test)
                cost = linear_regression.loss_(y_test, y_hat)
                model["train_cost"] = cost
                y_hat = linear_regression.predict_(x_test)
                cost = linear_regression.loss_(y_test, y_hat)
                model["test_cost"] = cost

                print(model)
                print()
                models.append(model)

                # Plot the results on 3 graphs (one for each feature)
                fig, ax = plt.subplots(1, 3)
                real_weight = x_test[:, 0]
                real_prod_distance = x_test[:, 1]
                real_time_delivery = x_test[:, 2]
                ax[0].scatter(real_weight, y_test, color="green")
                ax[0].scatter(x_test[:, 0], y_test, color="blue")
                ax[0].scatter(x_test[:, 0], y_hat, color="red")
                ax[0].set_title("Weight")
                ax[1].scatter(x_test[:, 1], y_test, color="blue")
                ax[1].scatter(x_test[:, 1], y_hat, color="red")
                ax[1].set_title("Product distance")
                ax[2].scatter(x_test[:, 2], y_test, color="blue")
                ax[2].scatter(x_test[:, 2], y_hat, color="red")
                ax[2].set_title("Time delivery")
                plt.show()

    best_model = min(models, key=lambda x: x["test_cost"])

    # Save the models in the file models.json
    with open("models.json", "w") as file:
        for model in models:
            file.write(json.dumps(model))
            if model != models[-1]:
                file.write(",\n")

    print("Best model is:")
    print(best_model)

    # plot the results on 3 graphs (one for each feature) for the best model
    degree = best_model["degres"][0]
    degree2 = best_model["degres"][1]
    degree3 = best_model["degres"][2]

    model_weight = polynomial_weight[:, :degree]
    model_prod_distance = polynomial_prod_distance[:, :degree2]
    model_time_delivery = polynomial_time_delivery[:, :degree3]

    x_train = numpy.concatenate(
        (model_weight[:, :best_model["degres"][0]],
            model_prod_distance[:, :best_model["degres"][1]],
            model_time_delivery[:, :best_model["degres"][2]]),
        axis=1)
    
    fig, ax = plt.subplots(3, 1)
    ax[0].scatter(x_test[:, 0], y_test, color="blue")
    ax[0].scatter(x_test[:, 0], y_hat, color="red")
    ax[0].set_title("Weight")
    ax[1].scatter(x_test[:, 1], y_test, color="blue")
    ax[1].scatter(x_test[:, 1], y_hat, color="red")
    ax[1].set_title("Product distance")
    ax[2].scatter(x_test[:, 2], y_test, color="blue")
    ax[2].scatter(x_test[:, 2], y_hat, color="red")
    ax[2].set_title("Time delivery")
    plt.show()


    # Test end
    
#    for feature_index in range(nb_features):
#
#        fix, ax = plt.subplots(1, 4)
#        costs = []
#
#        for degree in range(max_degree):
#
#            # Use your polynomial_features method on your training set
#            x_train_poly = add_polynomial_features(x_train[:, feature_index],
#                                                   degree + 1)
#
#            # Fit the model
#            linear_regression.thetas = numpy.ones((degree + 2, 1))
#            linear_regression.fit_(x_train_poly, y_train)
#
#            # Use the cost function to evaluate the model on the test set
#            x_test_poly = add_polynomial_features(x_test[:, feature_index],
#                                                  degree + 1)
#            y_hat = linear_regression.predict_(x_test_poly)
#            cost = linear_regression.loss_(y_hat, y_test)
#            costs.append(cost)
#            print("Cost for degree " + str(degree) + " : " + str(cost))
#
#            x_denormalized = linear_regression.denormalize(
#                x_test[:, feature_index])
#
#            ax[degree].scatter(x_denormalized, y_test, color="blue")
#            ax[degree].scatter(x_denormalized, y_hat, color="red")
#            ax[degree].set_title(
#                "Degree " + str(degree + 1) + "\n Cost : " + str(cost))
#
#        plt.suptitle(features[feature_index])
#        plt.show()
#
#        # Plot the cost depending on the degree
#        plt.figure()
#        plt.bar(range(1, max_degree + 1), costs)
#        plt.xlabel("Degree")
#        plt.ylabel("Cost")
#        plt.show()
#
#        print("\n\nBest degree for feature {}: ".format(
#            features[feature_index])
#              + str(costs.index(min(costs)) + 1))
#        print()
