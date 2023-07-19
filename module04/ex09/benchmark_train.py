import pandas
import numpy as np
from my_logistic_regression import MyLogisticRegression
import matplotlib.pyplot as plt

def get_dataset(path, features) -> pandas.DataFrame:

    try:
        dataset = pandas.read_csv(path)[features]
        if dataset.shape[0] == 0 or dataset.shape[1] != len(features):
            print("Error: The dataset is empty or has a wrong shape")
            exit(1)
        elif not all(feature in dataset.columns for feature in features):
            print("Error: The dataset is missing one or more features")
            exit(1)
        return dataset

    except Exception as e:
        print(e)
        print("Error: Cannot load the dataset")
        exit(1)


def split_dataset(dataset: pandas.DataFrame, ratios: float) -> tuple:

    try:
        # Shuffle the dataset
        dataset = dataset.sample(frac=1)

        # Training set = set of data that is used to train and
        # make the model learn
        m = dataset.shape[0]
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


def normalize_train(training: np.ndarray) -> tuple:
    try:
        if not isinstance(training, np.ndarray):
            print("Error: training must be a numpy.ndarray")
            exit(1)
        elif training.shape[0] == 0:
            print("Error: training must not be empty")
            exit(1)
        min = []
        max = []
        normalized = np.empty(training.shape)
        for i in range(training.shape[1]):
            min.append(np.min(training[:, i]))
            max.append(np.max(training[:, i]))
            normalized[:, i] = \
                (training[:, i] - min[i]) / (max[i] - min[i])
        return (normalized, min, max)
    except Exception:
        print("Error: Can't normalize the training dataset")
        exit(1)


def normalize_test(test: np.ndarray, min: list, max: list) \
        -> np.ndarray:
    try:
        if not isinstance(test, np.ndarray):
            print("Error: test must be a numpy.ndarray")
            exit(1)
        elif test.shape[0] == 0:
            print("Error: test must not be empty")
            exit(1)
        elif not isinstance(min, list) or not isinstance(max, list):
            print("Error: min and max must be lists")
            exit(1)
        elif len(min) != test.shape[1] or len(max) != test.shape[1]:
            print("Error: min and max must have the same size as test")
            exit(1)
        normalized = np.empty(test.shape)
        for i in range(test.shape[1]):
            normalized[:, i] = (test[:, i] - min[i]) / (max[i] - min[i])
        return normalized
    except Exception:
        print("Error: Can't normalize the test dataset")
        exit(1)


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
        if not isinstance(x, np.ndarray):
            return None
        m, n = x.shape
        if m == 0 or n == 0:
            return None
        if not isinstance(power, int) or power < 1:
            return None
        polynomial_matrix = x
        for i in range(2, power + 1):
            new_column = x ** i
            polynomial_matrix = np.c_[polynomial_matrix, new_column]
        return polynomial_matrix
    except Exception:
        return None


def confusion_matrix_(y_true, y_hat, labels=None, df_option=False):
    """
        Compute confusion matrix to evaluate the accuracy of a classification.
        Args:
            y: a numpy.array for the correct labels
            y_hat: a numpy.array for the predicted labels
            labels: optional, a list of labels to index the matrix.
                    This may be used to reorder or select a subset of labels.
                    (default=None)
            df_option: optional, if set to True the function will return a
                       pandas DataFrame instead of a numpy array.
                       (default=False)
        Return:
            The confusion matrix as a numpy array or a pandas DataFrame
            according to df_option value.
            None if any error.
        Raises:
            This function should not raise any Exception.
    """

    try:
        if not isinstance(y_true, np.ndarray) \
                or not isinstance(y_hat, np.ndarray):
            print("Not a numpy array")
            return None
        if y_true.shape != y_hat.shape:
            print("Shape error")
            return None
        if y_true.size == 0 or y_hat.size == 0:
            print("Empty array")
            return None
        if labels is None:
            labels = np.unique(np.concatenate((y_true, y_hat)))
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for i in range(len(labels)):
            for j in range(len(labels)):
                cm[i, j] = np.where((y_true == labels[i])
                                    & (y_hat == labels[j]))[0].shape[0]
        if df_option:
            cm = pandas.DataFrame(cm, index=labels, columns=labels)
        return cm

    except Exception:
        return None


if __name__ == "__main__":

    features = ["height", "weight", "bone_density"]
    target = ["Origin"]
    origins = {
        0: "Venus",
        1: "Earth",
        2: "Mars",
        3: "Belt Asteroids"
    }

    # Load the datasets
    x = get_dataset("../ressources/solar_system_census.csv", features)
    y = get_dataset("../ressources/solar_system_census_planets.csv", target)
    dataset = pandas.concat([x, y], axis=1)

    # Split the dataset into a training, a cross-validation and a test sets.
    (x_train, y_train,
     x_val, y_val,
     x_test, y_test) = split_dataset(dataset, (0.8, 0.1, 0.1))

    # Normalize the training features and se the same parameters
    # to normalize the validation and the test features.
    x_train_normalized, x_min, x_max = normalize_train(x_train)
    x_val_normalized = normalize_test(x_val, x_min, x_max)
    x_test_normalized = normalize_test(x_test, x_min, x_max)

    # Train different regularized logistic regression models with
    # a polynomial hypothesis of degree 3.
    polynomial_hypothesis = 3
    x_train_poly = add_polynomial_features(
        x_train_normalized, polynomial_hypothesis)
    x_val_poly = add_polynomial_features(
        x_val_normalized, polynomial_hypothesis)
    x_test_poly = add_polynomial_features(
        x_test_normalized, polynomial_hypothesis)

    theta_shape = (x.shape[1] * polynomial_hypothesis + 1, 1)

    # The models will be trained with different λ values, ranging from 0 to 1.
    lambdas = np.linspace(0.0, 1.0, num=6)
    for lambda_ in lambdas:

        print(f"One vs. all for lambda : {lambda_:.1f}")

        # Train 4 logistic regression to discriminate each planet
        models = []
        for current_train in range(len(origins)):

            print(f"Training model {current_train + 1} / 4 ... " +
                  f"From {origins[current_train]} ?")

            model = {}
            model["name"] = f"D{3}F{current_train}L{lambda_}"
            logistic_regression = MyLogisticRegression(
                theta=np.zeros(theta_shape),
                alpha=2.5,
                max_iter=50_000,
                lambda_=lambda_
            )
            current_y = np.where(y_train == current_train, 1, 0)
            model["theta"] = logistic_regression.fit_(x_train_poly, current_y)

            # Plot the loss evolution :
            # plt.plot(logistic_regression.losses)
            # plt.title(f"Loss evolution for {model['name']}")
            # plt.xlabel("Epochs")
            # plt.ylabel("Loss")
            # plt.show()

            models.append(model)

        # Evaluate the model with f1-score on the cross-validation set
        # For each element of the dataset
        nb_elmts = y_train.shape[0]
        y_predictions = np.empty((nb_elmts, 1))
        for i in range(nb_elmts):
            # Compute the probability of being from each planet
            y_probas = np.zeros((len(origins), 1))
            for current_test in range(len(origins)):
                current_lr = MyLogisticRegression(
                    theta=models[current_test]["theta"],
                    alpha=1,
                    max_iter=50_000,
                    lambda_=lambda_
                )
                current_x = x_train_poly[i].reshape(1, -1)
                proba = current_lr.predict_(current_x)
                y_probas[current_test] = proba
            y_predictions[i] = y_probas.argmax()

        confusion_matrix = \
            confusion_matrix_(y_train, y_predictions,
                              df_option=True, labels=[0, 1, 2, 3])
        print("Confusion matrix :\n", confusion_matrix)

        f1_score = logistic_regression.f1_score_(y_train, y_predictions)
        print("F1-score :", f1_score)

        accuracy = logistic_regression.accuracy_score_(y_train, y_predictions)
        print("Accuracy :", accuracy)
        print()
