from matplotlib import pyplot as plt
import pandas
import numpy
import yaml
from ridge import MyRidge


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

    def get_dataset(path: str, features: list) -> pandas.DataFrame:
        try:
            dataset = pandas.read_csv(path)[features]
            if dataset.shape[0] == 0 or dataset.shape[1] != 4:
                print("Error: The dataset is empty or has a wrong shape")
                exit(1)
            elif not all(_ in dataset.columns
                         for _ in features):
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

    # Load the dataset
    dataset_path = "../ressources/space_avocado.csv"
    features = ["weight", "prod_distance", "time_delivery"]
    target = ["target"]
    dataset = get_dataset(dataset_path, features + target)

    def split_dataset(dataset: pandas.DataFrame, ratios: float) -> tuple:
        try:

            # Shuffle the dataset
            dataset = dataset.sample(frac=1)

            m = dataset.shape[0]

            def normalize(x: numpy.ndarray) -> tuple:
                x_means = x.mean(axis=0)
                x_stds = x.std(axis=0)
                x_norm = (x - x_means) / x_stds
                return (x_norm, x_means, x_stds)

            # Normalize the dataset
            dataset, min_, max_ = normalize(dataset)
            dataset = pandas.DataFrame(dataset, columns=features + target)

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

    # Split your dataset into a training, a cross-validation and a test sets.
    (x_train, y_train,
     x_val, y_val,
     x_test, y_test) = split_dataset(dataset, (0.6, 0.2, 0.2))

    # Plot bar chart of the models cost
    costs = [model["cost"] for model in models]
    names = [model["name"] for model in models]
    plt.bar(names, costs)
    plt.xticks(rotation=90)
    plt.ylabel("Cost")
    plt.xlabel("Model name")
    plt.title("Comparaison of the models based on their cost " +
              "(lower is better)")
    plt.show()

    # Load the best model variables
    sorted_models = sorted(models, key=lambda k: k['cost'])
    best_model = sorted_models[0]

    name = best_model["name"]
    degree = best_model["degree"]
    lambda_ = best_model["lambda_"]
    print(f"Best model: {name} (degree={degree}, lambda={lambda_})")

    # Get the models of the best model degree
    degree_models = [model
                     for model in models
                     if model["degree"] == degree]
    degree_models.remove(best_model)

    # Instantiate the ridge model
    theta_shape = (x_train.shape[1] * degree + 1, 1)

    learning_rate = 10e-4
    n_cycle = 50_000
    ridge = MyRidge(numpy.zeros(theta_shape),
                    alpha=learning_rate,
                    max_iter=n_cycle)

    # Add polynomial features to the dataset with the best degree
    print(f"Adding polynomial features with degree {degree}...")
    x_train_degree = ridge.add_polynomial_features(x_train, degree)
    x_val_degree = ridge.add_polynomial_features(x_val, degree)
    x_test_degree = ridge.add_polynomial_features(x_test, degree)

    x_train_degree = ridge.add_polynomial_features(x_train, degree)
    x_val_degree = ridge.add_polynomial_features(x_val, degree)
    x_test_degree = ridge.add_polynomial_features(x_test, degree)

    model = {}
    model["name"] = f"D{degree}L{lambda_:.1f}"
    model["degree"] = degree
    print(f"Training model {model['name']}")

    # ###################################### #
    # Initialize the model's hyperparameters #
    # ###################################### #

    theta = numpy.zeros(theta_shape)
    ridge = MyRidge(theta, alpha=learning_rate,
                    max_iter=n_cycle, lambda_=lambda_)

    # ##################################### #
    # Train the model with the training set #
    # ##################################### #

    ridge.thetas = ridge.fit_(x_train_degree, y_train)

    # ########################################## #
    # Evaluate the model with the validation set #
    # ########################################## #

    y_hat = ridge.predict_(x_test_degree)
    cost = ridge.loss_(y_test, y_hat)

    model["theta"] = ridge.thetas
    model["y_hat"] = y_hat
    model["cost"] = cost

    print()

    # Plot the true price and the predicted price obtain
    # via your best model for each features with the different λ values
    # (meaning the dataset + the 5 predicted curves).

    features = ["weight", "prod_distance", "time_delivery"]
    target = ["target"]

    # Subplot for each feature
    fig, axs = plt.subplots(1, 3, figsize=(20, 10))

    for i, feature in enumerate(features):

        # Plot the dataset
        axs[i].scatter(x_test_degree[:, i], y_test, label="Dataset")

        # Plot the trained best model
        axs[i].scatter(x_test_degree[:, i], y_hat, label="Prediction")

        # Plot the pre-trained models of the same degree, but different lambda_
        for model_ in degree_models:
            axs[i].scatter(model_["x_val"][:, i], model_["y_hat"],
                           label=model_["name"], marker='.')

        axs[i].set_xlabel(feature)
        axs[i].set_ylabel("price")
        axs[i].legend()

    plt.show()

    # Two 4D plots to visualize the model prediction and the real values
    fig, ax = plt.subplots(1, 2,
                           figsize=(15, 5),
                           subplot_kw={"projection": "3d"})

    fig.colorbar(ax[0].scatter(x_test_degree[:, 0],
                               x_test_degree[:, 1],
                               x_test_degree[:, 2],
                               c=y_test), ax=ax[0], label="Price")

    fig.colorbar(ax[1].scatter(x_test_degree[:, 0],
                               x_test_degree[:, 1],
                               x_test_degree[:, 2],
                               c=y_hat), ax=ax[1], label="Price")

    ax[0].set_title("Real")
    ax[1].set_title("Predicted")

    for i in range(2):
        ax[i].set_xlabel(features[0])
        ax[i].set_ylabel(features[1])
        ax[i].set_zlabel(features[2])

    plt.show()