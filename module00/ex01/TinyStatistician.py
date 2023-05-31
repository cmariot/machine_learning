from numpy import array as nparray


class TinyStatistician:

    def __check_args__(self, array: nparray) -> bool:
        pass

    def mean(x: 'list[nparray]') -> float:
        if (not isinstance(x, (list, nparray))
                or len(x) == 0):
            return None
        total: float = 0.0
        for i in x:
            if not isinstance(i, (float, int)):
                return None
            total += i
        return total / len(x)

    def median(x):
        if not isinstance(x, (list, nparray)):
            return None
        elif not all(isinstance(elem, (int, float)) for elem in x):
            return None
        list_len = len(x)
        if list_len == 0:
            return None
        x.sort()
        if list_len % 2 == 0:
            return float(x[len(x) // 2 - 1] + x[len(x) // 2]) / 2
        return float(x[len(x) // 2])

    def quartile(x):
        if not isinstance(x, (list, nparray)):
            return None
        elif not all(isinstance(elem, (int, float)) for elem in x):
            return None
        list_len = len(x)
        if list_len == 0:
            return None
        x.sort()
        if list_len % 4 == 0:

            # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] -> Len = 12
            # [x, x, x, x, x, x, M, x, x, x,  x,  x] -> q1 = list at pos[3] -> Mean between value at pos 2 and 3




            q1 = float(x[list_len // 4 - 1] + x[list_len // 4]) / 2
            q3 = float(x[list_len // 2 - 1] + x[list_len // 2]) / 2
        else:
            # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] -> Len = 10
            # [x, x, Q, Q, x, M, x, Q, Q, x] -> q1 = list at pos[2.5] -> Mean between value at pos 2 and 3
            q1 = float(x[len(x) // 2])
            q3 = float(x[len(x) // 2])
        return [q1, q3]

    def percentile(x, p):
        if not isinstance(x, (list, nparray)):
            return None
        pass

    def var(x):
        if not isinstance(x, (list, nparray)):
            return None
        pass

    def std(x):
        if not isinstance(x, (list, nparray)):
            return None
        pass
