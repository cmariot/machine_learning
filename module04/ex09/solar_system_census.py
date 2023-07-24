import pickle
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
import pandas
from my_logistic_regression import MyLogisticRegression as MyLR
from benchmark_train import (get_dataset,
                             normalize_train,
                             normalize_test,
                             add_polynomial_features,
                             confusion_matrix_)


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
                x_test.to_numpy(), y_test.to_numpy(),
                x_validation.to_numpy(), y_validation.to_numpy())

    except Exception as e:
        print("Error: Can't split the dataset")
        print(e)
        exit(1)


def load_models(path):
    try:
        with open(path, "rb") as f:
            models = pickle.load(f)
        return models
    except Exception as e:
        print(e)
        exit(1)


def plot_models_scores(models: list):
    try:
        plt.bar(
            [model["lambda"] for model in models],
            [model["f1_score"] for model in models],
            width=0.1,
            color="green",
        )
        plt.xlabel("λ")
        plt.ylabel("F1 score")
        plt.title("Models scores")
        plt.show()
    except Exception as e:
        print(e)
        exit(1)


def get_best_model(models: list):
    try:
        return max(models, key=lambda x: x["f1_score"])
    except Exception as e:
        print(e)
        exit(1)


if __name__ == "__main__":

    # Loads the differents models from models.pickle and
    # train from scratch only the best one on a training set.
    models = load_models("models.pickle")

    # Visualize the performance of the different models with a bar plot
    # showing the score of the models given their λ value.
    plot_models_scores(models)

    # Train the best model on the training set
    best_model = get_best_model(models)

    # Load the dataset
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
     x_test, y_test) = split_dataset(dataset, (0.8, 0, 0.2))

    # Normalize the training features and use the same parameters
    # to normalize the validation and the test features.
    x_train_normalized, x_min, x_max = normalize_train(x_train)
    x_test_normalized = normalize_test(x_test, x_min, x_max)

    # Train different regularized logistic regression models with
    # a polynomial hypothesis of degree 3.
    polynomial_hypothesis = 3
    x_train_poly = add_polynomial_features(
        x_train_normalized, polynomial_hypothesis)
    x_test_poly = add_polynomial_features(
        x_test_normalized, polynomial_hypothesis)

    theta_shape = (x.shape[1] * polynomial_hypothesis + 1, 1)

    # Train the model with the best λ value on the training set
    # and evaluate its performance on the test set.
    lambda_ = best_model["lambda"]
    print(f"Training the best model with λ = {lambda_}")

    trained_thetas = []
    for i in range(len(origins)):

        print(f"Training model for {origins[i]}")

        mylr = MyLR(np.zeros(theta_shape),
                    alpha=1,
                    max_iter=100_000,
                    lambda_=lambda_)

        current_y_train = np.where(y_train == i, 1, 0)
        theta = mylr.fit_(x_train_poly, current_y_train)
        trained_thetas.append(theta)

        # Print the f1 score of all the models calculated on the test set.
        y_hat = mylr.predict_(x_test_poly)
        y_hat = np.where(y_hat >= 0.3, 1, 0)
        current_y_test = np.where(y_test == i, 1, 0)
        f1_score = mylr.f1_score_(current_y_test, y_hat)
        print(f"f1 score: {f1_score}")

    all_x = np.concatenate((x_train_poly, x_test_poly))
    all_x_denormalized = np.concatenate((x_train, x_test))
    all_y = np.concatenate((y_train, y_test))
    nb_elmts = all_y.shape[0]
    y_predictions = np.empty((nb_elmts, 1))
    for i in range(nb_elmts):
        # Compute the probability of being from each planet
        y_probas = np.zeros((len(origins), 1))
        for current_test in range(len(origins)):
            current_lr = MyLR(
                theta=trained_thetas[current_test],
                alpha=1,
                max_iter=50_000,
                lambda_=lambda_
            )
            current_x = all_x[i].reshape(1, -1)
            proba = current_lr.predict_(current_x)
            y_probas[current_test] = proba
        y_predictions[i] = y_probas.argmax()

    # Confusion matrix
    confusion_matrix = confusion_matrix_(all_y, y_predictions,
                                         labels=[0, 1, 2, 3],
                                         df_option=True)
    print(confusion_matrix)

    # Visualize the target values and the predicted values of the best model
    # on the same scatterplot.
    # Make some effort to have a readable figure.
    fig, ax = plt.subplots(1, 3)

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
        all_y < 2,
        np.where(
            all_y == 0,
            'green',    # 0
            'blue'      # 1
        ),
        np.where(
            all_y == 2,
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
            all_x_denormalized[:, index],
            all_x_denormalized[:, index + 1],
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
