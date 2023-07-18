import shutil
import sys
import argparse
import time
from matplotlib import patches
import pandas
import numpy as np
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

    def split_dataframes(features_dataframe: pandas.DataFrame,
                         target_dataframe: pandas.DataFrame,
                         ratio: float,
                         zipcode: int):
        try:
            if ratio < 0 or ratio > 1:
                print("Error: ratio must be between 0 and 1")
                exit(1)
            elif features_dataframe.shape[0] != target_dataframe.shape[0]:
                print("Error: The dataset and the target don't" +
                      " have the same number of elements")
                exit(1)

            complete_df = pandas.concat([features_dataframe, target_dataframe],
                                        axis=1, sort=False)

            # We shuffle the dataframe
            complete_df = complete_df.sample(frac=1).reset_index(drop=True)

            # Update the Origin column to be 1 if the planet is from the
            # specified zipcode, 0 otherwise
            complete_df["Origin"] = \
                complete_df["Origin"].apply(lambda x: 1 if x == zipcode else 0)

            # We split the dataframe into two dataframes
            # according to the ratio
            split_index = int(complete_df.shape[0] * ratio)
            features_train = complete_df.iloc[:split_index, :-1].to_numpy()
            features_test = complete_df.iloc[split_index:, :-1].to_numpy()
            target_train = complete_df.iloc[:split_index, -1:].to_numpy()
            target_test = complete_df.iloc[split_index:, -1:].to_numpy()

            return features_train, features_test, target_train, target_test

        except Exception:
            print("Error: Can't split the dataset")
            exit(1)

    features_train, features_test, target_train, target_test = \
        split_dataframes(features_dataframe, target_dataframe, 0.8, zipcode)

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

    # Normalize the training features
    x_train_normalized, x_min, x_max = normalize_train(features_train)
    x_test_normalized = normalize_test(features_test, x_min, x_max)

    class MyLogisticRegression():
        """
            Description:
                My personnal logistic regression to classify things.
        """

        def __init__(self, theta, alpha=0.001, max_iter=1000):

            if not isinstance(theta, np.ndarray):
                return None
            if not isinstance(alpha, (int, float)):
                return None
            if not isinstance(max_iter, int):
                return None
            if max_iter <= 0:
                return None
            if alpha <= 0:
                return None

            self.alpha = alpha
            self.max_iter = max_iter
            self.theta = theta

        @staticmethod
        def sigmoid_(x):
            """
            Compute the sigmoid of a vector.
            Args:
                x: has to be a numpy.ndarray.
            Returns:
                The sigmoid value as a numpy.ndarray.
                None if an exception occurs.
            Raises:
                This function should not raise any Exception.
            """
            try:
                return 1 / (1 + np.exp(-x))
            except Exception:
                return None

        def predict_(self, x):
            """
            Computes the vector of prediction y_hat from two
            non-empty numpy.ndarray.
            Args:
            x: has to be an numpy.ndarray, a vector of dimension m * n.
            Returns:
            y_hat as a numpy.ndarray, a vector of dimension m * 1.
            None if x or theta are empty numpy.ndarray.
            None if x or theta dimensions are not appropriate.
            Raises:
            This function should not raise any Exception.
            """

            if not isinstance(x, np.ndarray):
                return None

            try:
                m, n = x.shape

                if m == 0 or n == 0:
                    return None
                elif self.theta.shape != (n + 1, 1):
                    return None

                X_prime = np.hstack((np.ones((m, 1)), x))
                y_hat = self.sigmoid_(X_prime.dot(self.theta))
                return y_hat

            except Exception:
                return None

        def loss_elem_(self, y, y_hat, eps=1e-15):
            try:
                if not all(isinstance(x, np.ndarray) for x in [y, y_hat]):
                    return None

                m, n = y.shape
                if (m == 0 or n == 0):
                    return None
                elif y_hat.shape != (m, n):
                    return None

                dot1 = y * (np.log(y_hat + eps))
                dot2 = (1 - y) * np.log(1 - y_hat + eps)
                return -(dot1 + dot2)

            except Exception:
                return None

        def loss_(self, y, y_hat, eps=1e-15):
            """
                Compute the logistic loss value.
                Args:
                    y: has to be an numpy.ndarray, a vector of shape m * 1.
                    y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
                    eps: epsilon (default=1e-15)
                Returns:
                    The logistic loss value as a float.
                    None on any error.
                Raises:
                    This function should not raise any Exception.
            """

            try:
                loss_elem = self.loss_elem_(y, y_hat, eps)
                if loss_elem is None:
                    return None
                return np.mean(loss_elem)

            except Exception:
                return None

        def ft_progress(self, iterable,
                        length=shutil.get_terminal_size().columns - 2,
                        fill='█',
                        empty='░',
                        print_end='\r'):
            total = len(iterable)
            start = time.time()
            print()
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
            print()

        def gradient_(self, x, y):
            """
            Computes a gradient vector from three non-empty numpy.ndarray,
            without any for-loop.
            The three arrays must have compatible shapes.
            Args:
                x: has to be an numpy.ndarray, a matrix of shape m * n.
                y: has to be an numpy.ndarray, a vector of shape m * 1.
            Returns:
                The gradient as a numpy.ndarray, a vector of shape n * 1,
                    containg the result of the formula for all j.
                None if x, y, or theta are empty numpy.ndarray.
                None if x, y and theta do not have compatible shapes.
            Raises:
                This function should not raise any Exception.
            """

            try:
                if not all(isinstance(arr, np.ndarray) for arr in [x, y]):
                    return None

                m, n = x.shape

                if m == 0 or n == 0:
                    return None
                elif y.shape != (m, 1) or self.theta.shape != ((n + 1), 1):
                    return None

                X_prime = np.hstack((np.ones((m, 1)), x))
                y_hat = self.predict_(x)
                if y_hat is None:
                    return None
                return (X_prime.T.dot(y_hat - y)) / m

            except Exception:
                return None

        def fit_(self, x, y):
            for arr in [x, y]:
                if not isinstance(arr, np.ndarray):
                    print('x and y must be numpy.ndarray')
                    return None
            m, n = x.shape
            if m == 0 or n == 0:
                return None
            if y.shape != (m, 1):
                return None
            elif self.theta.shape != ((n + 1), 1):
                return None
            for _ in self.ft_progress(range(self.max_iter)):
                gradient = self.gradient_(x, y)
                if gradient is None:
                    return None
                self.theta = self.theta - self.alpha * gradient
            return self.theta

    theta = np.zeros((x_train_normalized.shape[1] + 1, 1))
    logistic_regression = MyLogisticRegression(
        theta,
        max_iter=150_000,
        alpha=1)
    logistic_regression.fit_(x_train_normalized, target_train)

    # Plot 3 scatter plots (one for each pair of citizen features) with
    # the dataset and the final prediction of the model.
    # The points must be colored following the real class of the citizen.
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    features_pairs = [
        ('weight', 'height'),
        ('weight', 'bone_density'),
        ('height', 'bone_density')
    ]

    # Group the test and the train sets together to plot them.
    features = np.vstack((x_train_normalized, x_test_normalized))
    features_denormalized = np.vstack((features_train, features_test))
    target = np.vstack((target_train, target_test))

    y_hat = logistic_regression.predict_(features)
    y_hat = np.where(y_hat >= 0.5, 1, 0)

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

    accuracy_score = accuracy_score_(target, y_hat)
    print("Accuracy score: {} %".format(accuracy_score * 100))

    colors = np.where(
        y_hat == target,
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

    fig.suptitle('Logistic regression')

    for i in range(3):
        index = i if i != 2 else -1

        ax[i].scatter(
            features_denormalized[:, index],
            features_denormalized[:, index + 1],
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
        features_denormalized[:, 0],
        features_denormalized[:, 1],
        features_denormalized[:, 2],
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
