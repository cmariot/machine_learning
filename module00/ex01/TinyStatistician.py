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
        list_len: int = len(x)
        sum: float = 0.0
        for i in x:
            sum += i
        return float(sum / list_len)

    def median(self, x) -> float:
        """
        Computes the median of a given non-empty list of int or float.
        """
        if TinyStatistician.__invalid_input(x):
            return None
        list_len = len(x)
        x.sort()
        middle = list_len // 2
        if list_len % 2 == 0:
            return float(x[middle - 1] + x[middle]) / 2
        else:
            return float(x[middle])

    def quartile(self, x) -> 'list[float]':
        """
        Computes the quartiles Q1 and Q3 of a given non-empty list of
        int or float.
        """
        if TinyStatistician.__invalid_input(x):
            return None
        list_len = len(x)
        x.sort()
        first_quartile_index = list_len // 4
        return [float(x[first_quartile_index]),
                float(x[first_quartile_index * 3])]

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
