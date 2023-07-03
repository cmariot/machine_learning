import numpy as np
from sklearn.metrics import accuracy_score, precision_score, \
    recall_score, f1_score


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


def precision_score_(y, y_hat, pos_label=1):
    """
    Compute the precision score.
    Args:
        y: a numpy.ndarray for the correct labels
        y_hat: a numpy.ndarray for the predicted labels
        pos_label: str or int, the class on which to report
                   the precision_score (default=1)
    Return:
        The precision score as a float.
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

        if y.size == 0 or y_hat.size == 0:
            return None

        if not isinstance(pos_label, (int, str)):
            return None

        tp = np.sum(np.logical_and(y == pos_label, y == y_hat))
        fp = np.sum(np.logical_and(y != pos_label, y_hat == pos_label))
        return tp / (tp + fp)

    except Exception:
        return None


def recall_score_(y, y_hat, pos_label=1):
    """
    Compute the recall score.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
        pos_label: str or int, the class on which to report
                   the precision_score (default=1)
    Return:
        The recall score as a float.
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

        if y.size == 0 or y_hat.size == 0:
            return None

        if not isinstance(pos_label, (int, str)):
            return None

        tp = np.sum(np.logical_and(y == pos_label, y == y_hat))
        fn = np.sum(np.logical_and(y == pos_label, y_hat != pos_label))
        return tp / (tp + fn)

    except Exception:
        return None


def f1_score_(y, y_hat, pos_label=1):
    """
    Compute the f1 score.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
        pos_label: str or int, the class on which to report
                   the precision_score (default=1)
    Returns:
        The f1 score as a float.
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

        if y.size == 0 or y_hat.size == 0:
            return None

        if not isinstance(pos_label, (int, str)):
            return None

        precision = precision_score_(y, y_hat, pos_label)
        recall = recall_score_(y, y_hat, pos_label)
        return 2 * (precision * recall) / (precision + recall)

    except Exception:
        return None


if __name__ == "__main__":

    # Example 1:
    print("Example 1:")
    y_hat = np.array(
        [1, 1, 0, 1, 0, 0, 1, 1]).reshape((-1, 1))
    y = np.array(
        [1, 0, 0, 1, 0, 1, 0, 0]).reshape((-1, 1))

    # Accuracy
    print("Accuracy")
    # your implementation
    print(accuracy_score_(y, y_hat))
    # Output:
    # 0.5
    # sklearn implementation
    print(accuracy_score(y, y_hat))
    # Output:
    # 0.5

    # Precision
    print("Precision")
    # your implementation
    print(precision_score_(y, y_hat))
    # Output:
    # 0.4
    # sklearn implementation
    print(precision_score(y, y_hat))
    # Output:
    # 0.4

    # Recall
    print("Recall")
    # your implementation
    print(recall_score_(y, y_hat))
    # Output: 0.6666666666666666
    # sklearn implementation
    print(recall_score(y, y_hat))
    # Output: 0.6666666666666666

    # F1-score
    print("F1-score")
    # your implementation
    print(f1_score_(y, y_hat))
    # Output:
    # 0.5
    # sklearn implementation
    print(f1_score(y, y_hat))
    # Output:
    # 0.5

    # Example 2:
    print("\nExample 2:")
    y_hat = np.array(['norminet',
                      'dog',
                      'norminet',
                      'norminet',
                      'dog',
                      'dog',
                      'dog',
                      'dog'])
    y = np.array(['dog',
                  'dog',
                  'norminet',
                  'norminet',
                  'dog',
                  'norminet',
                  'dog',
                  'norminet'])

    # Accuracy
    print("Accuracy")
    # your implementation
    print(accuracy_score_(y, y_hat))
    # Output:
    # 0.625
    # sklearn implementation
    print(accuracy_score(y, y_hat))
    # Output:
    # 0.625

    # Precision
    print("Precision")
    # your implementation
    print(precision_score_(y, y_hat, pos_label='dog'))
    # Output:
    # 0.6
    # sklearn implementation
    print(precision_score(y, y_hat, pos_label='dog'))
    # Output:
    # 0.6

    # Recall
    print("Recall")
    # your implementation
    print(recall_score_(y, y_hat, pos_label='dog'))
    # Output:
    # 0.75
    # sklearn implementation
    print(recall_score(y, y_hat, pos_label='dog'))
    # Output:
    # 0.75

    # F1-score
    print("F1-score")
    # your implementation
    print(f1_score_(y, y_hat, pos_label='dog'))
    # Output:
    # 0.6666666666666665
    # sklearn implementation
    print(f1_score(y, y_hat, pos_label='dog'))
    # Output:
    # 0.6666666666666665

    # Example 3:
    print("\nExample 3:")
    y_hat = np.array(['norminet',
                      'dog',
                      'norminet',
                      'norminet',
                      'dog',
                      'dog',
                      'dog',
                      'dog'])
    y = np.array(['dog',
                  'dog',
                  'norminet',
                  'norminet',
                  'dog',
                  'norminet',
                  'dog',
                  'norminet'])

    # Precision
    print("Precision")
    # your implementation
    print(precision_score_(y, y_hat, pos_label='norminet'))
    # Output:
    # 0.6666666666666666
    # sklearn implementation
    print(precision_score(y, y_hat, pos_label='norminet'))
    # Output:
    # 0.6666666666666666

    # Recall
    print("Recall")
    # your implementation
    print(recall_score_(y, y_hat, pos_label='norminet'))
    # Output:
    # 0.5
    # sklearn implementation
    print(recall_score(y, y_hat, pos_label='norminet'))
    # Output:
    # 0.5

    # F1-score
    print("F1-score")
    # your implementation
    print(f1_score_(y, y_hat, pos_label='norminet'))
    # Output:
    # 0.5714285714285715
    # sklearn implementation
    print(f1_score(y, y_hat, pos_label='norminet'))
    # Output:
    # 0.5714285714285715
