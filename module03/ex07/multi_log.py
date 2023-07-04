from matplotlib import patches, pyplot as plt
import numpy as np
import pandas
import time
import shutil
import pandas as pd


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
                print(1)
                return None
            elif self.theta.shape != (n + 1, 1):
                print(2)
                return None

            X_prime = np.hstack((np.ones((m, 1)), x))
            y_hat = self.sigmoid_(X_prime.dot(self.theta))
            if y_hat is None:
                print(3)
            return y_hat

        except Exception as e:
            print("Exception !")
            print(e)
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


if __name__ == "__main__":

    # Split the dataset into a training and a test set.
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
                         ratio: float):
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

            print(complete_df)

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
        split_dataframes(features_dataframe, target_dataframe, 0.8)

    # Normalize the training and the test features,
    # keep the min/max for the denormalization
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

    trained_thetas = []

    for current_train in range(4):

        print("Training model {} / 4 ...".format(current_train + 1))

        # Train 4 logistic regression classifiers to discriminate each class
        # from the others
        theta = np.zeros((x_train_normalized.shape[1] + 1, 1))
        logistic_regression = MyLogisticRegression(theta, 1, 50_000)

        y: np.ndarray = target_train.copy()

        # Set the target at 1 if equal to current_train else 0
        y = np.where(y == current_train, 1, 0)

        theta = logistic_regression.fit_(x_train_normalized, y)

        trained_thetas.append(theta)

    # Predict for each example the class according to each classifiers
    # and select the one with the highest output probability.

    normalized_features = np.concatenate(
        (x_test_normalized, x_train_normalized),
        axis=0)
    denormalized_features = np.concatenate(
        (features_test, features_train),
        axis=0)
    total_target = np.concatenate((target_test, target_train), axis=0)
    y_predictions = np.empty((normalized_features.shape[0], 1))

    for i in range(normalized_features.shape[0]):
        y_proba = np.empty((4, 1))
        for current_test in range(4):
            logistic_regression.theta = trained_thetas[current_test]
            y_proba[current_test] = \
                logistic_regression.predict_(normalized_features[i].reshape(
                    1, normalized_features.shape[1])
                )
        y_predictions[i] = y_proba.argmax()

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

    accuracy_score = accuracy_score_(total_target, y_predictions)
    print("Accuracy score: {} %".format(accuracy_score * 100))

    def confusion_matrix_(y_true, y_hat, labels=None, df_option=False):
        """
            Compute confusion matrix to evaluate the accuracy
            of a classification.
            Args:
                y: a numpy.array for the correct labels
                y_hat: a numpy.array for the predicted labels
                labels: optional, a list of labels to index the matrix.
                        This may be used to reorder or select a subset of
                        labels. (default=None)
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
                cm = pd.DataFrame(cm, index=labels, columns=labels)

            return cm

        except Exception:
            return None

    print("Confusion matrix:")
    print(confusion_matrix_(total_target, y_predictions, df_option=True))

    # Plot 3 scatter plots (one for each pair of citizen features)
    # with the dataset and the final prediction of the model.
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    colors = np.where(
        y_predictions < 2,
        np.where(
            y_predictions == 0,
            'green',    # 0
            'blue'      # 1
        ),
        np.where(
            y_predictions == 2,
            'orange',   # 2
            'red'       # 3
        )
    )

    real_colors = np.where(
        total_target < 2,
        np.where(
            total_target == 0,
            'green',    # 0
            'blue'      # 1
        ),
        np.where(
            total_target == 2,
            'orange',   # 2
            'red'       # 3
        )
    )

    features_pairs = [
        ('weight', 'height'),
        ('weight', 'bone_density'),
        ('height', 'bone_density')
    ]

    for i in range(3):
        index = i if i != 2 else -1

        ax[i].scatter(
            denormalized_features[:, index],
            denormalized_features[:, index + 1],
            c=colors.flatten(),
            marker='o',
            alpha=0.5,
            edgecolors=real_colors.flatten()
        )

        ax[i].set_xlabel(features_pairs[i][0])
        ax[i].set_ylabel(features_pairs[i][1])
        ax[i].set_title(f'{features_pairs[i][1]} vs {features_pairs[i][0]}')

    fig.legend(
        handles=[
            patches.Patch(
                          color='green',
                          label='The flying cities of Venus'),
            patches.Patch(
                          color='blue',
                          label='United Nations of Earth'),
            patches.Patch(
                          color='orange',
                          label='Mars Republic'),
            patches.Patch(
                          color='red',
                          label="The Asteroid's Belt colonies"),
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
        denormalized_features[:, 0],
        denormalized_features[:, 1],
        denormalized_features[:, 2],
        c=colors.flatten(),
        marker='o',
        alpha=0.5,
        edgecolors=real_colors.flatten()
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
