import numpy as np
import unittest


def sigmoid_(x):
    """
    Compute the sigmoid of a vector.
    Args:
        x: has to be a numpy.ndarray of shape (m, 1).
    Returns:
        The sigmoid value as a numpy.ndarray of shape (m, 1).
        None if x is an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    try:
        if not isinstance(x, np.ndarray):
            return None
        m = x.shape[0]
        if m == 0 or x.shape != (m, 1):
            return None
        return 1.0 / (1.0 + np.exp(-x))
    except Exception:
        return None


class TestSigmoid(unittest.TestCase):

    def test_sigmoid_with_non_array_input(self):
        x = 5
        self.assertIsNone(sigmoid_(x))

    def test_sigmoid_with_empty_input(self):
        x = np.array([])
        self.assertIsNone(sigmoid_(x))

    def test_sigmoid_with_wrong_shape_input(self):
        x = np.array([[-4, 2], [0, 1]])
        self.assertIsNone(sigmoid_(x))

    def test_sigmoid_with_one_sample(self):
        x = np.array([[-4]])
        self.assertEqual(sigmoid_(x), 0.01798620996209156)

    def test_sigmoid_with_one_sample_high_value(self):
        x = np.array([[2]])
        self.assertEqual(sigmoid_(x), 0.8807970779778823)

    def test_sigmoid_with_multiple_samples(self):
        x = np.array([[-4], [2], [0]])
        expected_output = np.array([[0.01798620996209156],
                                    [0.8807970779778823],
                                    [0.5]])
        self.assertTrue(np.array_equal(sigmoid_(x), expected_output))


if __name__ == "__main__":

    x = np.array([[-4]])
    print(sigmoid_(x))
    # 0.01798620996209156

    x = np.array([[2]])
    print(sigmoid_(x))
    # 0.8807970779778823

    x = np.array([[-4], [2], [0]])
    print(sigmoid_(x))
    # array([[0.01798621],
    #        [0.88079708],
    #        [0.5       ]])

    unittest.main()
