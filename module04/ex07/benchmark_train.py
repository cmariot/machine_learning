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
                         for _ in ["weight", "prod_distance",
                                   "time_delivery", "target"]):
                print("Error: The dataset is missing one or more features")
                exit(1)
            print(dataset)
            print(dataset.describe())
            print()
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

    models = []
    ridge = MyRidge(numpy.zeros((2, 1)))
    lambdas = numpy.arange(0.0, 1.2, 0.2)

    # For each degree of the polynomial features
    for degree in range(1, 5):

        x_train_degree = ridge.add_polynomial_features(x_train, degree)
        x_test_degree = ridge.add_polynomial_features(x_test, degree)

        def normalize(x: numpy.ndarray) -> tuple:
            x_means = numpy.mean(x, axis=0)
            x_stds = numpy.std(x, axis=0)
            x_norm = (x - x_means) / x_stds
            return (x_norm, x_means, x_stds)

        x_train_degree_norm, x_means, x_stds = normalize(x_train_degree)
        x_test_degree_norm = (x_test_degree - x_means) / x_stds

        for lambda_ in lambdas:

            model = {}
            model["name"] = f"D{degree}L{lambda_:.1f}"
            print(f"Training model {model['name']}")

            # ##################################### #
            # Train the model with the training set #
            # ##################################### #

            # Train the model with the training set
            learning_rate = 10e-4
            n_cycle = 10_000
            theta = numpy.zeros((x_train_degree_norm.shape[1] + 1, 1))

            ridge = MyRidge(theta, learning_rate, n_cycle, lambda_)

            ridge.thetas = ridge.fit_(x_train_degree_norm, y_train)
            model["theta"] = ridge.thetas

            # #################################### #
            # Evaluate the model with the test set #
            # #################################### #

            y_hat = ridge.predict_(x_test_degree_norm)
            cost = ridge.loss_(y_test, y_hat)
            print(f"cost = {cost}")
            model["cost"] = cost

            models.append(model)
            print()

            # ######################################### #
            # Plot the model's predictions with the test #
            # ######################################### #

            # Plot the model's predictions with the test set
            for i in range(3):
                plt.scatter(x_test_degree_norm[:, i], y_test, color="blue")
                plt.scatter(x_test_degree_norm[:, i], y_hat, color="red")
                plt.title(f"Model {model['name']}")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.legend(["y", "y_hat"])
                plt.show()

            # ############################### #
            # Plot the model's cost evolution #
            # ############################### #

            # Plot the model's cost evolution
            plt.plot(ridge.losses)
            plt.title(f"Model {model['name']}")
            plt.xlabel("n_cycle")
            plt.ylabel("cost")
            plt.show()

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
    for model in models[:5]:
        print(f"- {model['name']} : {model['cost']}")

    # Save the models in the file "models.yml"
    with open("models.yml", "w") as file:
        yaml.dump(models, file)

    print("Models saved in the file \"models.yml\"")

    print("Done")
