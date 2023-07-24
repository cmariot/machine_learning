from matplotlib import pyplot as plt
import pandas
import numpy
import yaml
from ridge import MyRidge


def get_models(path):
    """
    Load the trained models
    """
    try:
        with open(path, "r") as f:
            models = yaml.load(f, Loader=yaml.loader.UnsafeLoader)
            return models

    except Exception:
        print("Error loading the smodels.yml file.\n" +
              "Please train the models first.\n" +
              "> python3 benchmark_train.py")
        exit(1)


def get_dataset(path: str, features: list) -> pandas.DataFrame:
    """
    Load the dataset
    """
    try:
        dataset = pandas.read_csv(path)[features]
        if dataset.shape[0] == 0 or dataset.shape[1] != 4:
            print("Error: The dataset is empty or has a wrong shape")
            exit(1)
        return dataset
    except Exception as e:
        print(e)
        exit(1)


def split_dataset(dataset: pandas.DataFrame, ratios: float) -> tuple:
    try:

        if sum(ratios) != 1.0:
            raise Exception("Error: split_dataset's ratios sum must be 1.")

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


def plot_models_costs(models):
    """
    Plot bar chart of the models cost
    """
    try:
        costs = [model["cost"] for model in models]
        names = [model["name"] for model in models]
        plt.bar(names, costs)
        plt.xticks(rotation=90)
        plt.ylabel("Cost")
        plt.xlabel("Model name")
        plt.title("Comparaison of the models based on their cost " +
                "(lower is better)")
        plt.show()
    except Exception:
        print("Error: Can't plot the model costs")
        exit()


def get_best_model(models):
    try:
        sorted_models = sorted(models, key=lambda k: k['cost'])
        best_model = sorted_models[0]
        name = best_model["name"]
        degree = best_model["degree"]
        lambda_ = best_model["lambda_"]
        print(f"Best model: {name} (degree={degree}, lambda={lambda_})")
        return (name, degree, lambda_)
    except Exception:
        print("Error: Model selection failed.")
        exit()


if __name__ == "__main__":

    # Load the yml file with the benchmark parameters
    models = get_models("models.yml")
    name, degree, lambda_ = get_best_model(models)
    trained_models = [model for model in models if (model["degree"] == degree and model["lambda_"] != lambda_)]

    plot_models_costs(models)

    # Load the dataset
    features = ["weight", "prod_distance", "time_delivery"]
    target = ["target"]
    dataset = get_dataset("../ressources/space_avocado.csv", features + target)

    # Split your dataset into a training, a cross-validation and a test sets.
    (x_train, y_train,
     x_val, y_val,
     x_test, y_test) = split_dataset(dataset, (0.6, 0.2, 0.2))

    # Normalize features of the training and the test sets,
    # save variables for denormalization
    def normalize_train(x: numpy.ndarray) -> tuple:
        x_means = x.mean(axis=0)
        x_stds = x.std(axis=0)
        x_norm = (x - x_means) / x_stds
        return (x_norm, x_means, x_stds)

    def normalize(x, x_means, x_stds):
        return (x - x_means) / x_stds

    # Normalize the dataset
    x_train_norm, x_means, x_stds = normalize_train(x_train)
    x_val_norm = normalize(x_val, x_means, x_stds)
    x_test_norm = normalize(x_test, x_means, x_stds)

    # ###################################### #
    # Initialize the model's hyperparameters #
    # ###################################### #

    theta_shape = (x_train.shape[1] * degree + 1, 1)
    learning_rate = 10e-4
    n_iter = 30_000
    ridge = MyRidge(numpy.zeros(theta_shape),
                    alpha=learning_rate,
                    max_iter=n_iter,
                    lambda_=lambda_)

    # Add polynomial features to the dataset with the best degree
    x_train_degree = ridge.add_polynomial_features(x_train_norm, degree)
    x_val_degree = ridge.add_polynomial_features(x_val_norm, degree)
    x_test_degree = ridge.add_polynomial_features(x_test_norm, degree)

    # ##################################### #
    # Train the model with the training set #
    # ##################################### #

    ridge.thetas = ridge.fit_(x_train_degree, y_train)

    # ########################################## #
    # Evaluate the model with the validation set #
    # ########################################## #

    y_hat = ridge.predict_(x_val_degree)
    cost = ridge.loss_(y_val, y_hat)

    # Plot the true price and the predicted price obtain
    # via your best model for each features with the different λ values
    # (meaning the dataset + the 5 predicted curves).

    # Subplot for each feature
    fig, axs = plt.subplots(1, 3, figsize=(20, 10))

    # Concatenate the validation and the test sets
    x = numpy.concatenate((x_val_degree, x_test_degree))
    y = numpy.concatenate((y_val, y_test))
    y_hat = ridge.predict_(x)

    for i, feature in enumerate(features):

        axs[i].scatter(x[:, i], y, label="Dataset")

        # Plot the trained best model
        axs[i].scatter(x[:, i], y_hat, label="Prediction")

        axs[i].set_xlabel(feature)
        axs[i].set_ylabel("Price")


    # Plot the pre-trained models of the same degree, but different lambda_
    for model_ in trained_models:
        print(model_["name"])
        model_ridge = MyRidge(model_["theta"], alpha=learning_rate,
                                max_iter=n_iter, lambda_=model_["lambda_"])
        print(model_ridge.theta)
        #model_ridge.fit_(x_train_degree, y_train)
        model_y_hat = model_ridge.predict_(x)

        for i, feature in enumerate(features):
            axs[i].scatter(x[:, i], model_y_hat,
                            label=model_["name"], marker='.')

    for i in range(len(features)):
        axs[i].legend()

    plt.show()