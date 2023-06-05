import unittest


class TinyStatistician:

    @staticmethod
    def __invalid_input(x):
        if not isinstance(x, list):
            return True
        elif not all(isinstance(i, (int, float)) for i in x):
            return True
        elif len(x) == 0:
            return True
        return False

    def mean(self, x) -> float:
        """
        Computes the mean of a given non-empty list of int or float.
        """
        if TinyStatistician.__invalid_input(x):
            return None
        total = 0
        for i in x:
            total += i
        return float(total / len(x))

    def median(self, x) -> float:
        """
        Computes the median of a given non-empty list of int or float.
        """
        if TinyStatistician.__invalid_input(x):
            return None
        x.sort()
        middle = len(x) // 2
        if len(x) % 2 == 0:
            return (x[middle - 1] + x[middle]) / 2
        else:
            return float(x[middle])

    def quartile(self, x) -> 'list[float]':
        """
        Computes the quartiles Q1 and Q3 of a given non-empty list of
        int or float.
        """
        if TinyStatistician.__invalid_input(x):
            return None
        x.sort()
        middle = len(x) // 2
        Q1_index = middle // 2
        Q3_index = middle + Q1_index
        if len(x) % 2 == 0:
            Q1 = (x[Q1_index - 1] + x[Q1_index]) / 2
            Q3 = (x[Q3_index - 1] + x[Q3_index]) / 2
        else:
            Q1 = x[Q1_index]
            Q3 = x[Q3_index]
        return [float(Q1), float(Q3)]

    def percentile(self, x, percentile) -> float:
        """
        Computes the percentile p of a given non-empty list of int or float.
        Note:
        uses a different definition of percentile, it does linear
        interpolation between the two closest list element to the percentile.
        """
        if TinyStatistician.__invalid_input(x):
            return None
        elif not isinstance(percentile, int):
            return None
        elif percentile < 0 or percentile > 100:
            return None
        x.sort()
        list_len = len(x)
        index = int(percentile / 100 * (list_len - 1))
        if index == list_len - 1:
            return float(x[index])
        else:
            return float(x[index] + (x[index + 1] - x[index]) *
                         (percentile / 100 * (list_len - 1) - index))

    def var(self, x) -> float:
        """
        Computes the variance of a given non-empty list of int or float.
        Note:
        uses the unbiased estimator (divides by n - 1).
        """
        if TinyStatistician.__invalid_input(x):
            return None
        mean = TinyStatistician.mean(self, x)
        diff = sum((i - mean) ** 2 for i in x)
        return float(diff / (len(x) - 1))

    def std(self, x):
        """
        Computes the standard deviation of a given non-empty list of
        int or float.
        """
        var = TinyStatistician.var(self, x)
        if var is None:
            return None
        return var ** 0.5


class TestTinyStatistician(unittest.TestCase):
    def setUp(self):
        self.ts = TinyStatistician()

    def test_mean(self):
        self.assertAlmostEqual(
            self.ts.mean([1, 2, 3, 4, 5]), 3.0)
        self.assertAlmostEqual(
            self.ts.mean([1.5, 2.5, 3.5, 4.5, 5.5]), 3.5)
        self.assertAlmostEqual(
            self.ts.mean([1, 2, 3, 4, 5, 6]), 3.5)
        self.assertAlmostEqual(
            self.ts.mean([1.5, 2.5, 3.5, 4.5, 5.5, 6.5]), 4.0)
        self.assertAlmostEqual(
            self.ts.mean([1, 2, 3, 4, 5, 6, 7]), 4.0)
        self.assertAlmostEqual(
            self.ts.mean([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]), 4.5)
        self.assertAlmostEqual(
            self.ts.mean([1, 2, 3, 4, 5, 6, 7, 8]), 4.5)
        self.assertAlmostEqual(
            self.ts.mean([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]), 5.0)
        self.assertAlmostEqual(
            self.ts.mean([1, 2, 3, 4, 5, 6, 7, 8, 9]), 5.0)
        self.assertAlmostEqual(
            self.ts.mean([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]), 5.5)

    def test_mean_with_invalid_input(self):
        self.assertIsNone(self.ts.mean([]))
        self.assertIsNone(self.ts.mean([1, 2, 'a', 4, 5]))
        self.assertIsNone(self.ts.mean('1, 2, 3, 4, 5'))

    def test_median(self):
        self.assertAlmostEqual(
            self.ts.median([1, 2, 3, 4, 5]), 3.0)
        self.assertAlmostEqual(
            self.ts.median([1.5, 2.5, 3.5, 4.5, 5.5]), 3.5)
        self.assertAlmostEqual(
            self.ts.median([1, 2, 3, 4, 5, 6]), 3.5)
        self.assertAlmostEqual(
            self.ts.median([1.5, 2.5, 3.5, 4.5, 5.5, 6.5]), 4.0)
        self.assertAlmostEqual(
            self.ts.median([1, 2, 3, 4, 5, 6, 7]), 4.0)
        self.assertAlmostEqual(
            self.ts.median([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]), 4.5)
        self.assertAlmostEqual(
            self.ts.median([1, 2, 3, 4, 5, 6, 7, 8]), 4.5)
        self.assertAlmostEqual(
            self.ts.median([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]), 5.0)
        self.assertAlmostEqual(
            self.ts.median([1, 2, 3, 4, 5, 6, 7, 8, 9]), 5.0)
        self.assertAlmostEqual(
            self.ts.median([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]), 5.5)

    def test_median_with_invalid_input(self):
        self.assertIsNone(self.ts.median([]))
        self.assertIsNone(self.ts.median([1, 2, 'a', 4, 5]))
        self.assertIsNone(self.ts.median('1, 2, 3, 4, 5'))

    def test_quartile(self):
        self.assertEqual(
            self.ts.quartile([1, 42, 300, 10, 59]), [10.0, 59.0])
        self.assertEqual(
            self.ts.quartile([1, 2, 3, 4, 5, 6, 7, 8, 9]), [3.0, 7.0])
        self.assertEqual(
            self.ts.quartile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), [2.5, 7.5])
        self.assertEqual(
            self.ts.quartile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]), [3.0, 8.0])

    def test_quartile_with_invalid_input(self):
        self.assertIsNone(self.ts.quartile([]))
        self.assertIsNone(self.ts.quartile([1, 2, 'a', 4, 5]))
        self.assertIsNone(self.ts.quartile('1, 2, 3, 4, 5'))

    def test_percentile(self):
        self.assertAlmostEqual(
            self.ts.percentile([1, 2, 3, 4, 5], 10), 1.4)
        self.assertAlmostEqual(
            self.ts.percentile([1, 2, 3, 4, 5, 6], 10), 1.5)
        self.assertAlmostEqual(
            self.ts.percentile([1, 2, 3, 4, 5, 6, 7], 10), 1.6)
        self.assertAlmostEqual(
            self.ts.percentile([1, 2, 3, 4, 5, 6, 7, 8], 10), 1.7)
        self.assertAlmostEqual(
            self.ts.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9], 10), 1.8)

    def test_percentile_with_invalid_input(self):
        self.assertIsNone(
            self.ts.percentile([], 10))
        self.assertIsNone(
            self.ts.percentile([1, 2, 'a', 4, 5], 10))
        self.assertIsNone(
            self.ts.percentile('1, 2, 3, 4, 5', 10))
        self.assertIsNone(
            self.ts.percentile([1, 2, 3, 4, 5], 'a'))
        self.assertIsNone(
            self.ts.percentile([1, 2, 3, 4, 5], -10))
        self.assertIsNone(
            self.ts.percentile([1, 2, 3, 4, 5], 110))

    def test_var(self):
        self.assertAlmostEqual(
            self.ts.var([1, 2, 3, 4, 5]), 2.5)
        self.assertAlmostEqual(
            self.ts.var([1.5, 2.5, 3.5, 4.5, 5.5]), 2.5)
        self.assertAlmostEqual(
            self.ts.var([1, 2, 3, 4, 5, 6]), 3.5)
        self.assertAlmostEqual(
            self.ts.var([1.5, 2.5, 3.5, 4.5, 5.5, 6.5]), 3.5)

    def test_var_with_invalid_input(self):
        self.assertIsNone(self.ts.var([]))
        self.assertIsNone(self.ts.var([1, 2, 'a', 4, 5]))
        self.assertIsNone(self.ts.var('1, 2, 3, 4, 5'))

    def test_std(self):
        self.assertAlmostEqual(
            self.ts.std([1, 2, 3, 4, 5]), 1.5811, places=4)
        self.assertAlmostEqual(
            self.ts.std([1.5, 2.5, 3.5, 4.5, 5.5]), 1.5811, places=4)
        self.assertAlmostEqual(
            self.ts.std([1, 2, 3, 4, 5, 6]), 1.8708, places=4)
        self.assertAlmostEqual(
            self.ts.std([1.5, 2.5, 3.5, 4.5, 5.5, 6.5]), 1.8708, places=4)
        self.assertAlmostEqual(
            self.ts.std([1, 2, 3, 4, 5, 6, 7]), 2.1602, places=4)
        self.assertAlmostEqual(
            self.ts.std([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]), 2.1602, places=4)
        self.assertAlmostEqual(
            self.ts.std([1, 2, 3, 4, 5, 6, 7, 8]), 2.4495, places=4)
        self.assertAlmostEqual(
            self.ts.std([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]),
            2.4495, places=4)
        self.assertAlmostEqual(
            self.ts.std([1, 2, 3, 4, 5, 6, 7, 8, 9]), 2.7386, places=4)
        self.assertAlmostEqual(
            self.ts.std([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]),
            2.7386, places=4)

    def test_std_with_invalid_input(self):
        self.assertIsNone(self.ts.std([]))
        self.assertIsNone(self.ts.std([1, 2, 'a', 4, 5]))
        self.assertIsNone(self.ts.std('1, 2, 3, 4, 5'))


if __name__ == "__main__":
    a = [1, 42, 300, 10, 59]
    print(TinyStatistician().mean(a))
    # Output:
    # 82.4
    print(TinyStatistician().median(a))
    # Output:
    # 42.0
    print(TinyStatistician().quartile(a))
    # Output:
    # [10.0, 59.0]
    print(TinyStatistician().percentile(a, 10))
    # Output:
    # 4.6
    print(TinyStatistician().percentile(a, 15))
    # Output:
    # 6.4
    print(TinyStatistician().percentile(a, 20))
    # Output:
    # 8.2
    print(TinyStatistician().var(a))
    # Output:
    # 15349.3
    print(TinyStatistician().std(a))
    # Output:
    # 123.89229193133849

    unittest.main()
