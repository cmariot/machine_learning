import shutil
import time
from matplotlib import pyplot as plt
import pandas
import numpy
import yaml


if __name__ == "__main__":

    # Load the yml file with the benchmark parameters
    try:
        with open("models.yml", "r") as f:
            models = yaml.load(f, Loader=yaml.loader.UnsafeLoader)
    except Exception:
        print("Error loading the smodels.yml file.")
        print("Please train the models first.")
        print("> python3 benchmark_train.py")
        exit(1)

    def check_models():
        """Check if the models are trained."""
        if models is None:
            print("Please train the models first.")
            exit(1)
        if not isinstance(models, list):
            print("Please train the models first.")
            exit(1)
        elif not all(isinstance(model, dict) for model in models):
            print("Please train the models first.")
            exit(1)
        if len(models) == 0:
            print("Please train the models first.")
            exit(1)
        if not all("name" in model for model in models):
            print("Please train the models first.")
            exit(1)
        if not all("cost" in model for model in models):
            print("Please train the models first.")
            exit(1)
        if not all("degree" in model for model in models):
            print("Please train the models first.")
            exit(1)
        if not all("theta" in model for model in models):
            print("Please train the models first.")
            exit(1)

    check_models()

    # Get the model with the lowest cost
    best_model = min(models, key=lambda x: x["cost"])

    print("The model we will train is : {}".format(best_model["name"]))

    def get_dataset(path: str, features: list) -> pandas.DataFrame:
        try:
            dataset = pandas.read_csv(path)[features]
            if dataset.shape[0] == 0 or dataset.shape[1] != 4:
                print("Error: The dataset is empty or has a wrong shape")
                exit(1)
            elif not all(_ in dataset.columns
                         for _ in ["weight", "prod_distance",
                                   "time_delivery", "target"]):
                print("Error: The dataset is missing one or more features")
                exit(1)
            print(dataset)
            print(dataset.describe())
            return dataset
        except Exception:
            print("Error: Can't find the dataset file")
            exit(1)

    # Load the dataset
    dataset_path = "../ressources/space_avocado.csv"
    dataset = \
        get_dataset(dataset_path,
                    ["weight", "prod_distance", "time_delivery", "target"])
    features = ["weight", "prod_distance", "time_delivery"]
    target = ["target"]

    def split_dataset(dataset: pandas.DataFrame, proportion: float) -> tuple:
        if not isinstance(proportion, (int, float)):
            print("Error: The proportion must be a number")
            exit(1)
        elif proportion < 0 or proportion > 1:
            print("Error: The proportion must be between 0 and 1")
            exit(1)
        try:
            dataset = dataset.sample(frac=1).reset_index(drop=True)
            m = dataset.shape[0]
            train_size = int(m * proportion)
            train_set = dataset[:train_size]
            if train_set.shape[0] == 0:
                print("Error: The train dataset is empty")
                exit(1)
            test_set = dataset[train_size:]
            if test_set.shape[0] == 0:
                print("Error: The test dataset is empty")
                exit(1)
            x_train = train_set[features].to_numpy()
            y_train = train_set[target].to_numpy()
            x_test = test_set[features].to_numpy()
            y_test = test_set[target].to_numpy()
            return (x_train, y_train, x_test, y_test)
        except Exception:
            print("Error: Can't split the dataset")
            exit(1)

    # Shuffle and split the dataset into training and test sets
    x_train, y_train, x_test, y_test = split_dataset(dataset, 0.8)

    def normalize_train(training: numpy.ndarray) -> tuple:
        try:
            if not isinstance(training, numpy.ndarray):
                print("Error: training must be a numpy.ndarray")
                exit(1)
            elif training.shape[0] == 0:
                print("Error: training must not be empty")
                exit(1)
            min = []
            max = []
            normalized = numpy.empty(training.shape)
            for i in range(training.shape[1]):
                min.append(numpy.min(training[:, i]))
                max.append(numpy.max(training[:, i]))
                normalized[:, i] = \
                    (training[:, i] - min[i]) / (max[i] - min[i])
            return (normalized, min, max)
        except Exception:
            print("Error: Can't normalize the training dataset")
            exit(1)

    def normalize_test(test: numpy.ndarray, min: list, max: list) \
            -> numpy.ndarray:
        try:
            if not isinstance(test, numpy.ndarray):
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
            normalized = numpy.empty(test.shape)
            for i in range(test.shape[1]):
                normalized[:, i] = (test[:, i] - min[i]) / (max[i] - min[i])
            return normalized
        except Exception:
            print("Error: Can't normalize the test dataset")
            exit(1)

    # Normalize the training features
    x_train_normalized, x_min, x_max = normalize_train(x_train)

    # Normalize the training target
    y_train_normalized, y_min, y_max = normalize_train(y_train)

    # Normalize the test set using the same normalization parameters
    # as the training set
    x_test_normalized = normalize_test(x_test, x_min, x_max)
    y_test_normalized = normalize_test(y_test, y_min, y_max)

    def add_polynomial_features(x, degree):
        """
        x = numpy.array([1, 2, 3])
        x_poly = add_polynomial_features(x, 3)
        print(x_poly)
        -> array([[ 1., 1., 1., 2., 4., 8., 3., 9., 27.]])
        """
        if not isinstance(degree, int):
            print("Error: The degree must be an integer")
            exit(1)
        elif degree < 0:
            print("Error: The degree must be positive")
            exit(1)
        if degree == 0:
            return numpy.ones((x.shape[0], 1))
        res = numpy.empty((x.shape[0], x.shape[1] * degree))
        for i in range(x.shape[1]):
            for j in range(degree):
                res[:, i * degree + j] = x[:, i] ** (j + 1)
        return res

    # Add polynomial features to the training and the test set
    x_train_poly = add_polynomial_features(x_train_normalized, 4)
    x_test_poly = add_polynomial_features(x_test_normalized, 4)

    def predict_(x: numpy.ndarray, theta: numpy.ndarray) -> numpy.ndarray:
        try:
            if not isinstance(x, numpy.ndarray) \
                    or not isinstance(theta, numpy.ndarray):
                print("Error: x and theta must be numpy.ndarray")
                exit(1)
            elif x.shape[1] != theta.shape[0]:
                print("Error: theta must have the same size as x columns")
                exit(1)
            return numpy.dot(x, theta)
        except Exception:
            print("Error: Can't predict the dataset")
            exit(1)

    def ft_progress(iterable,
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

    def fit_(x: numpy.ndarray, y: numpy.ndarray, theta: numpy.ndarray,
             alpha: float, max_iter: int) -> tuple:
        try:
            if not isinstance(x, numpy.ndarray) \
                    or not isinstance(y, numpy.ndarray):
                print("Error: x and y must be numpy.ndarray")
                exit(1)
            elif x.shape[0] != y.shape[0]:
                print("Error: x and y must have compatible shapes")
                exit(1)
            elif not isinstance(theta, numpy.ndarray):
                print("Error: theta must be a numpy.ndarray")
                exit(1)
            elif theta.shape[0] != x.shape[1]:
                print(theta.shape[0], x.shape[1])
                exit(1)
            elif not isinstance(alpha, (int, float)):
                print("Error: alpha must be a number")
                exit(1)
            elif alpha < 0:
                print("Error: alpha must be positive")
                exit(1)
            elif not isinstance(max_iter, int):
                print("Error: max_iter must be an integer")
                exit(1)
            elif max_iter < 0:
                print("Error: max_iter must be positive")
                exit(1)
            for i in ft_progress(range(max_iter)):
                y_hat = predict_(x, theta)
                gradient = numpy.dot(x.T, y_hat - y) / x.shape[0]
                theta -= (alpha * gradient)
            return theta
        except Exception:
            print("Error: Can't fit the model")
            exit(1)

    def get_model_x(x_train_poly, degree):
        weight = x_train_poly[:, 0:degree[0]]
        distance = x_train_poly[:, 4:4 + degree[1]]
        time = x_train_poly[:, 8:8 + degree[2]]
        model_x = numpy.concatenate((weight, distance, time), axis=1)
        return model_x

    # Get the training model features
    train_model_x = get_model_x(x_train_poly, best_model["degree"])

    # Train the model with the training set
    learning_rate = 10e-7
    n_cycle = 100_000
    theta = numpy.array(best_model["theta"]).reshape(-1, 1)

    print(f"Training model {best_model['name']} ... ", end="")

    theta = fit_(
        train_model_x, y_train_normalized,
        theta, learning_rate, n_cycle)

    # #################################### #
    # Evaluate the model with the test set #
    # #################################### #

    test_model_x = get_model_x(x_test_poly, best_model["degree"])
    y_hat = predict_(test_model_x, theta)

    def cost_elem_(y: numpy.ndarray, y_hat: numpy.ndarray) \
            -> numpy.ndarray:
        try:
            if not isinstance(y, numpy.ndarray) \
                    or not isinstance(y_hat, numpy.ndarray):
                print("Error: y and y_hat must be numpy.ndarray")
                exit(1)
            elif y.shape != y_hat.shape:
                print("Error: y and y_hat must have the same shape")
                exit(1)
            return ((y_hat - y) ** 2) / (2 * y.shape[0])
        except Exception:
            print("Error: Can't compute the cost")
            exit(1)

    def cost_(y: numpy.ndarray, y_hat: numpy.ndarray) -> float:
        try:
            if not isinstance(y, numpy.ndarray) \
                    or not isinstance(y_hat, numpy.ndarray):
                print("Error: y and y_hat must be numpy.ndarray")
                exit(1)
            elif y.shape != y_hat.shape:
                print("Error: y and y_hat must have the same shape")
                exit(1)
            return sum(cost_elem_(y, y_hat))[0]
        except Exception:
            print("Error: Can't compute the cost")
            exit(1)

    cost = cost_(y_test_normalized, y_hat)
    print(f"Done !\nCost = {cost}")

    # ########################################### #
    # Plot the model prediction with the test set #
    # ########################################### #

    def denormalize_prediction(y_hat, y_min, y_max):
        return y_hat * (y_max - y_min) + y_min

    y_hat_denormalized = denormalize_prediction(y_hat, y_min[0], y_max[0])

    fig, ax = plt.subplots(3, 3, figsize=(10, 10))

    for i in range(3):

        for j in range(3):

            if j == 0:
                ax[i, j].scatter(x_test[:, i],
                                 y_test,
                                 color="blue", alpha=0.5)
            elif j == 1:
                ax[i, j].scatter(x_test[:, i],
                                 y_hat_denormalized,
                                 color="red", alpha=0.5)
            elif j == 2:
                ax[i, j].scatter(x_test[:, i],
                                 y_test,
                                 color="blue", alpha=0.5)
                ax[i, j].scatter(x_test[:, i],
                                 y_hat_denormalized,
                                 color="red", alpha=0.5)
            ax[i, j].set_xlabel(features[i])
            ax[i, j].set_ylabel("Price")

    cols = ["Real values", "Predicted values", "Both"]

    for ax, col in zip(ax[0], cols):
        ax.set_title(col)

    plt.tight_layout()
    plt.show()

    # Two 4D plots to visualize the model prediction and the real values
    fig, ax = plt.subplots(1, 2,
                           figsize=(15, 5),
                           subplot_kw={"projection": "3d"})

    fig.colorbar(ax[0].scatter(x_test[:, 0], x_test[:, 1], x_test[:, 2],
                               c=y_test), ax=ax[0], label="Price")
    fig.colorbar(ax[1].scatter(x_test[:, 0], x_test[:, 1], x_test[:, 2],
                               c=y_hat_denormalized), ax=ax[1], label="Price")

    ax[0].set_title("Real")
    ax[1].set_title("Predicted")

    for i in range(2):
        ax[i].set_xlabel(features[0])
        ax[i].set_ylabel(features[1])
        ax[i].set_zlabel(features[2])

    plt.show()
