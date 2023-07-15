import pandas
import numpy
import yaml
from ridge import MyRidge
import matplotlib.pyplot as plt


if __name__ == "__main__":

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
    # Shuffle and split the dataset into training and test sets
    (x_train, y_train,
     x_val, y_val,
     x_test, y_test) = split_dataset(dataset, (0.6, 0.2, 0.2))

    models = []
    ridge = MyRidge(numpy.zeros((2, 1)))
    lambdas = numpy.arange(0.0, 1.2, 0.2)

    # For each degree of the polynomial features
    for degree in range(1, 5):

        x_train_degree = ridge.add_polynomial_features(x_train, degree)
        x_val_degree = ridge.add_polynomial_features(x_val, degree)
        x_test_degree = ridge.add_polynomial_features(x_test, degree)

        for lambda_ in lambdas:

            model = {}
            model["degree"] = degree
            model["lambda_"] = lambda_
            model["name"] = f"D{degree}L{lambda_:.1f}"

            print(f"Training model {model['name']}")

            # ##################################### #
            # Train the model with the training set #
            # ##################################### #

            # Train the model with the training set
            learning_rate = 10e-5
            n_cycle = 200_000
            theta = numpy.zeros((x_train_degree.shape[1] + 1, 1))

            ridge = MyRidge(theta, learning_rate, n_cycle, lambda_)

            ridge.thetas = ridge.fit_(x_train_degree, y_train)
            model["theta"] = ridge.thetas

            # ########################################## #
            # Evaluate the model with the validation set #
            # ########################################## #

            y_hat = ridge.predict_(x_val_degree)
            cost = ridge.loss_(y_val, y_hat)
            print(f"cost = {cost}")
            model["cost"] = cost

            models.append(model)
            print()

            # # ######################################### #
            # # Plot the model's predictions with the test #
            # # ######################################### #

            # # Plot the model's predictions with the test set
            # for i in range(3):
            #     plt.scatter(x_test_degree[:, i], y_test, color="blue")
            #     plt.scatter(x_test_degree[:, i], y_hat, color="red")
            #     plt.title(f"Model {model['name']}")
            #     plt.xlabel("x")
            #     plt.ylabel("y")
            #     plt.legend(["y", "y_hat"])
            #     plt.show()

            # # # ############################### #
            # # # Plot the model's cost evolution #
            # # # ############################### #

            # # Plot the model's cost evolution
            # if lambda_ == 0.0:
            #     plt.plot(ridge.losses)
            #     plt.title(f"Model {model['name']}")
            #     plt.xlabel("n_cycle")
            #     plt.ylabel("cost")
            #     plt.show()

    # Plot bar chart of the models cost
    costs = [model["cost"] for model in models]
    names = [model["name"] for model in models]
    plt.bar(names, costs)
    plt.xticks(rotation=90)
    plt.ylabel("Cost")
    plt.xlabel("Model name")
    plt.title("Comparaison of the models based on"
              + " their cost (lower is better)")
    plt.show()

    # Sort the models by cost
    models = sorted(models, key=lambda k: k['cost'])

    # Print the best models
    print("Models sorted by cost :")
    for i, model in enumerate(models[:5]):
        print(f"{i + 1}- {model['name']} : {model['cost']}")

    # Evalue the best model with the test set
    best_model = models[0]
    print(f"\nEvaluating the best model {best_model['name']}\
        with the test set")
    degree = best_model["degree"]
    lambda_ = best_model["lambda_"]
    theta = best_model["theta"]
    ridge = MyRidge(theta, lambda_)
    x_test_degree = ridge.add_polynomial_features(x_test, degree)
    y_hat = ridge.predict_(x_test_degree)
    cost = ridge.loss_(y_test, y_hat)
    print(f"cost = {cost}")

    # Save the models in the file "models.yml"
    with open("models.yml", "w") as file:
        yaml.dump(models, file)

    print("Models saved in the file \"models.yml\"")

    print("Done")
