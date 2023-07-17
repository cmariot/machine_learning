import argparse
import sys
from matplotlib import patches
import numpy as np
import pandas
from my_logistic_regression import MyLogisticRegression
import matplotlib.pyplot as plt


if __name__ == "__main__":

    def parse_zipcode():
        """
        Parses the command-line arguments and returns the value
        of the --zipcode argument.

        Returns:
            The value of the --zipcode argument as an integer.
            None if --zipcode is not specified.
        """
        try:
            parser = argparse.ArgumentParser()
            parser.add_argument("-zipcode", type=int, help="zipcode",
                                choices=(0, 1, 2, 3), required=True)
            args = parser.parse_args()
            return args.zipcode
        except argparse.ArgumentError:
            print("Invalid argument: --zipcode must be 0, 1, 2, or 3")
            parser.print_usage(sys.stderr)
            return None

    zipcode = parse_zipcode()
    if zipcode is None:
        exit()

    def get_dataframe(path: str, features: list) -> pandas.DataFrame:
        try:
            dataset = pandas.read_csv(path)[features]
            if dataset.shape[0] == 0:
                print("Error: The dataset is empty")
                exit(1)
            elif dataset.shape[1] != len(features):
                print("Error: The dataset is missing one or more features")
                exit(1)
            return dataset
        except Exception:
            print("Error: Can't find the dataset file")
            exit(1)

    features_path = "../ressources/solar_system_census.csv"
    features_dataframe = \
        get_dataframe(features_path, ["weight", "height", "bone_density"])

    target_path = "../ressources/solar_system_census_planets.csv"
    target_dataframe = \
        get_dataframe(target_path, ["Origin"])

    dataset = pandas.concat([features_dataframe, target_dataframe], axis=1)
    features = ["weight", "height", "bone_density"]
    target = ["Origin"]

    def split_dataset(dataset: pandas.DataFrame, ratios: float) -> tuple:
        try:

            # Shuffle the dataset
            dataset = dataset.sample(frac=1)

            m = dataset.shape[0]

            # Set the origin value to 1 if the planet is from zipcode,
            # 0 otherwise
            y = np.where(dataset["Origin"] == zipcode, 1, 0)

            def normalize(x: np.ndarray) -> tuple:
                x_means = x.mean(axis=0)
                x_stds = x.std(axis=0)
                x_norm = (x - x_means) / x_stds
                return (x_norm, x_means, x_stds)

            # Normalize the dataset
            dataset, min_, max_ = normalize(dataset[features])

            # Concate the normalized dataset and the target
            dataset = pandas.concat(
                [dataset, pandas.DataFrame(y, columns=target)], axis=1)

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

        except Exception as e:
            print(e)
            print("Error: Can't split the dataset")
            exit(1)

    # Split your dataset into a training, a cross-validation and a test sets.
    # Shuffle and split the dataset into training and test sets
    (x_train, y_train,
     x_val, y_val,
     x_test, y_test) = split_dataset(dataset, (0.6, 0.2, 0.2))

    degree = 3
    lambdas = np.arange(0, 1.2, 0.2)
    models = []

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

    x_test_poly = add_polynomial_features(x_test, degree)
    x_train_poly = add_polynomial_features(x_train, degree)
    x_val_poly = add_polynomial_features(x_val, degree)

    models = []
    for lambda_ in lambdas:

        # Create the model
        model = {}

        model["degree"] = degree
        model["lambda"] = lambda_
        model["name"] = f"D{degree}L{lambda_}"

        logistic_regression = MyLogisticRegression(
            theta=np.zeros((degree * degree + 1, 1)),
            alpha=10e-3,
            max_iter=10_000,
            lambda_=lambda_
            )

        # Train the model
        logistic_regression.fit_(x_train_poly, y_train)
        model["theta"] = logistic_regression.theta

        # Predict the results
        y_hat = logistic_regression.predict_(x_val_poly)
        y_hat = np.where(y_hat >= 0.5, 1, 0)

        # Evaluate the model with the f1 score
        f1_score = logistic_regression.f1_score_(y_val, y_hat)
        model["f1_score"] = f1_score
        print(f"f1_score: {f1_score}")

        models.append(model)

        # Plot the model loss evolution
        plt.plot(logistic_regression.losses)
        plt.title(f"Model {model['name']} loss evolution")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.show()

    # Sort the models by f1_score
    models = sorted(models, key=lambda k: k["f1_score"], reverse=True)

    # Print the best model
    print(f"Best model: {models[0]}")

    # Save the best model
    np.save("models.npy", models)

    # Plot the best model
    logistic_regression = MyLogisticRegression(
        theta=models[0]["theta"],
        alpha=10e-4,
        max_iter=100_000,
        lambda_=models[0]["lambda"]
        )

    logistic_regression.fit_(x_train_poly, y_train)

    # Predict the results
    y_hat = logistic_regression.predict_(x_train_poly)
    y_hat = np.where(y_hat >= 0.5, 1, 0)

    # Plot 3 scatter plots (one for each pair of citizen features) with
    # the dataset and the final prediction of the model.
    # The points must be colored following the real class of the citizen.
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    features_pairs = [
        ('weight', 'height'),
        ('weight', 'bone_density'),
        ('height', 'bone_density')
    ]

    # Calculate and display the fraction of correct predictions over the total
    # number of predictions based on the test set.
    def accuracy_score_(y, y_hat):
        """
        Compute the accuracy score.
        Args:
            y: a numpy.ndarray for the correct labels
            y_hat:a numpy.ndarray for the predicted labels
        Returns:
            The accuracy score as a float.
            None on any error.
        Raises:
            This function should not raise any Exception.
        """

        try:
            if not isinstance(y, np.ndarray) \
                    or not isinstance(y_hat, np.ndarray):
                return None
            if y.shape != y_hat.shape:
                return None
            if y.size == 0:
                return None
            true = np.where(y == y_hat)[0].shape[0]
            return true / y.size

        except Exception:
            return None

    accuracy_score = accuracy_score_(y_train, y_hat)
    print("Accuracy score: {} %".format(accuracy_score * 100))

    colors = np.where(
        y_hat == y_train,
        np.where(
            y_hat == 1,
            'green',    # True positive
            'blue'      # True negative
        ),
        np.where(
            y_hat == 1,
            'orange',   # False positive
            'red'       # False negative
        )
    )

    fig.suptitle('Logistic regression\n')

    for i in range(3):
        index = i if i != 2 else -1

        ax[i].scatter(
            x_train_poly[:, index],
            y_train,
            c=colors.flatten(),
            marker='o',
            alpha=0.5,
            edgecolors='none'
        )

        ax[i].set_xlabel(features_pairs[i][0])
        ax[i].set_ylabel(features_pairs[i][1])
        ax[i].set_title(f'{features_pairs[i][1]} vs {features_pairs[i][0]}')

    fig.legend(
        handles=[
            patches.Patch(color='green',
                          label='True positive'),
            patches.Patch(color='blue',
                          label='True negative'),
            patches.Patch(color='orange',
                          label='False positive'),
            patches.Patch(color='red',
                          label='False negative'),
        ],
        loc='lower center', ncol=4, fontsize='small',
    )

    plt.show()

    # Plot a 3D scatter plot with the dataset and the final prediction
    # of the model. The points must be colored following the real class
    # of the citizen.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        x_train_poly[:, 0],
        x_train_poly[:, 1],
        x_train_poly[:, 2],
        c=colors.flatten(),
        marker='o',
        alpha=0.5,
        edgecolors='none'
    )

    ax.set_xlabel('weight')
    ax.set_ylabel('height')
    ax.set_zlabel('bone_density')
    ax.set_title('bone_density vs height vs weight')

    fig.legend(
        handles=[
            patches.Patch(color='green',
                          label='True positive'),
            patches.Patch(color='blue',
                          label='True negative'),
            patches.Patch(color='orange',
                          label='False positive'),
            patches.Patch(color='red',
                          label='False negative'),
        ],
        loc='lower center', ncol=4, fontsize='small',
    )

    plt.show()
