import pandas
import numpy
import yaml
from ridge import MyRidge
import matplotlib.pyplot as plt


def get_dataset(path: str, features: list) -> pandas.DataFrame:
    try:
        dataset = pandas.read_csv(path)[features]
        if dataset.shape[0] == 0 or dataset.shape[1] != 4:
            print("Error: The dataset is empty or has a wrong shape")
            exit(1)
        elif not all(_ in dataset.columns for _ in features):
            print("Error: The dataset is missing one or more features")
            exit(1)
        print("Dataset loaded.")
        print("\nDataset description :")
        print(dataset)
        print(dataset.describe())
        print()
        return dataset
    except Exception as e:
        print(e)
        exit(1)


def split_dataset(dataset: pandas.DataFrame, ratios: float) -> tuple:
    try:

        # Shuffle the dataset
        dataset = dataset.sample(frac=1)

        m = dataset.shape[0]

        # Training set = set of data that is used to train and
        # make the model learn
        train_index_begin = 0
        train_index_end = int(m * ratios[0])
        x_train = dataset[features][train_index_begin:train_index_end]
        y_train = dataset[target][train_index_begin:train_index_end]

        # Test set = set of data that is used to test the model
        test_index_begin = train_index_end
        test_index_end = test_index_begin + int(m * ratios[1])
        x_test = dataset[features][test_index_begin:test_index_end]
        y_test = dataset[target][test_index_begin:test_index_end]

        # Cross-validation set = set of data that is used to tune
        # the model's hyperparameters. The model is trained on the
        # training set,and, simultaneously, the model evaluation is
        # performed on the validation set after every epoch.
        val_index_begin = test_index_end
        val_index_end = val_index_begin + int(m * ratios[2])
        x_validation = dataset[features][val_index_begin:val_index_end]
        y_validation = dataset[target][val_index_begin:val_index_end]

        # Return the splitted dataset as Numpy arrays
        return (x_train.to_numpy(), y_train.to_numpy(),
                x_validation.to_numpy(), y_validation.to_numpy(),
                x_test.to_numpy(), y_test.to_numpy())

    except Exception:
        print("Error: Can't split the dataset")
        exit(1)


def normalize_train(x: numpy.ndarray) -> tuple:
    x_means = x.mean(axis=0)
    x_stds = x.std(axis=0)
    x_norm = (x - x_means) / x_stds
    return (x_norm, x_means, x_stds)


def normalize(x, x_means, x_stds):
    return (x - x_means) / x_stds


def add_polynomial_features(x, power):
    """
    Add polynomial features to matrix x by raising its columns
    to every power in the range of 1 up to the power given in argument.
    Args:
        x: has to be an numpy.ndarray, a matrix of shape m * n.
        power: has to be an int, the power up to which the columns
                of matrix x are going to be raised.
    Returns:
        The matrix of polynomial features as a numpy.ndarray,
            of shape m * (np), containg the polynomial feature values
            for all training examples.
        None if x is an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """

    try:
        if not isinstance(x, numpy.ndarray):
            return None
        m, n = x.shape
        if m == 0 or n == 0:
            return None
        if not isinstance(power, int) or power < 1:
            return None
        polynomial_matrix = x
        for i in range(2, power + 1):
            new_column = x ** i
            polynomial_matrix = numpy.c_[polynomial_matrix, new_column]
        return polynomial_matrix

    except Exception:
        return None


if __name__ == "__main__":

    # Load the dataset
    dataset_path = "../ressources/space_avocado.csv"
    features = ["weight", "prod_distance", "time_delivery"]
    target = ["target"]
    dataset = get_dataset(dataset_path, features + target)

    # Split your dataset into a training, a cross-validation and a test sets.
    (x_train, y_train,
     x_val, y_val,
     x_test, y_test) = split_dataset(dataset, (0.8, 0.1, 0.1))

    # Normalize features of the training and the test sets,
    # save variables for denormalization
    x_train_norm, x_means, x_stds = normalize_train(x_train)
    x_val_norm = normalize(x_val, x_means, x_stds)
    x_test_norm = normalize(x_test, x_means, x_stds)

    models = []
    lambdas = numpy.linspace(0.0, 1.0, 6)

    # For each degree of the polynomial features
    for degree in range(1, 5):

        x_train_degree = add_polynomial_features(x_train_norm, degree)
        x_val_degree = add_polynomial_features(x_val_norm, degree)
        x_test_degree = add_polynomial_features(x_test_norm, degree)

        for lambda_ in lambdas:

            model = {}
            model["name"] = f"D{degree}L{lambda_:.1f}"
            model["degree"] = degree
            model["lambda_"] = lambda_
            print(f"Training model {model['name']}")

            # ##################################### #
            # Train the model with the training set #
            # ##################################### #

            learning_rate = 10e-4
            n_cycle = 25_000
            theta = numpy.zeros((x_train_degree.shape[1] + 1, 1))
            ridge = MyRidge(theta, learning_rate, n_cycle, lambda_)
            ridge.theta = ridge.fit_(x_train_degree, y_train)
            model["theta"] = ridge.theta

            # ########################################## #
            # Evaluate the model with the validation set #
            # ########################################## #

            y_hat = ridge.predict_(x_val_degree)
            cost = ridge.loss_(y_val, y_hat)
            model["cost"] = cost
            print(f"cost (evaluated with validation set) : {cost}")

            # Save the model
            models.append(model)
            print()

            if lambda_ == 0.0:

                # ############################ #
                # Plot the model's predictions #
                # ############################ #

                for i in range(3):
                    plt.scatter(x_val[:, i], y_val, color="blue")
                    plt.scatter(x_val[:, i], y_hat, color="red")
                    plt.title(f"Model {model['name']}")
                    plt.xlabel("x")
                    plt.ylabel("y")
                    plt.legend(["real values", "predicted values"])
                    plt.show()

                # ############################### #
                # Plot the model's cost evolution #
                # ############################### #

                plt.plot(ridge.losses)
                plt.title(f"Model {model['name']} cost evolution")
                plt.xlabel("iteration")
                plt.ylabel("cost")
                plt.show()

    # Plot bar chart of the models cost
    # costs = [model["cost"] for model in models]
    # names = [model["name"] for model in models]
    # plt.bar(names, costs)
    # plt.xticks(rotation=90)
    # plt.ylabel("Cost")
    # plt.xlabel("Model name")
    # plt.title("Comparaison of the models based on their cost " +
    #           "(lower is better)")
    # plt.show()

    # Save the models in the file "models.yml"
    with open("models.yml", "w") as file:
        yaml.dump(models, file)
    print("Models saved in the file \"models.yml\"")

    # Sort the models by cost
    models = sorted(models, key=lambda k: k['cost'])

    # Print the best models
    print("\nModels sorted by cost :")
    for i, model in enumerate(models[:5]):
        print(f"{i + 1}- {model['name']} : {model['cost']}")

    # Make an hypothesis
    best_model = models[0]
    print("\nAccording to the model benchmark,",
          f"we gonna train the model {best_model['name']}",
          f"(degree {best_model['degree']},",
          f"lambda {best_model['lambda_']})")

    # Evalue the best model with the test set
    print(f"\nEvaluating the best model {best_model['name']}",
          "with the test set")

    ridge = MyRidge(theta=best_model["theta"],
                    alpha=10e-4,
                    max_iter=100_000,
                    lambda_=best_model["lambda_"])

    x_test_degree = add_polynomial_features(x_test_norm, best_model["degree"])
    y_hat = ridge.predict_(x_test_degree)
    cost = ridge.loss_(y_test, y_hat)
    print(f"cost (evaluated with validation set) : {best_model['cost']}")
    print(f"cost (evaluated with test set) : {cost}")
