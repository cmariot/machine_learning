class Matrix:

    # The matrix object can be initialized with either:
    # - the elements of the matrix as a list of lists
    #   Matrix([[1.0, 2.0], [3.0, 4.0]])
    # - the shape of the matrix as a tuple
    #   Matrix((2, 2)) -> [[0.0, 0.0], [0.0, 0.0]]
    def __init__(self,
                 data: 'list[list[float]]' = None,
                 shape: 'tuple[int, int]' = None):
        if data is None and shape is None:
            raise TypeError(
                "Matrix() missing 1 required argument : 'data' or 'shape'")
        elif data is not None and shape is not None:
            raise TypeError(
                "Matrix() takes 1 positional argument but 2 were given")
        if data is not None:
            self.__init_by_data(data)
        elif shape is not None:
            self.__init_by_shape(shape)

    def __init_by_data(self, data):
        if (not isinstance(data, list)
                or not all(isinstance(x, list) for x in data)):
            raise TypeError("Data must be a list of lists")
        elif len(data) == 0:
            raise ValueError("Data must not be empty")
        elif not all(len(x) == len(data[0]) for x in data):
            raise ValueError(
                "Data must be a matrix, all rows must have the same length")
        elif len(data[0]) == 0:
            raise ValueError("Data must not be empty")
        elif not all(
                isinstance(x, (int, float)) for row in data for x in row):
            raise TypeError("Data must contain only integers or floats")
        self.data = [[float(x) for x in row] for row in data]
        self.shape = (len(data), len(data[0]))
        if self.shape[0] == 1 or self.shape[1] == 1:
            self.__class__ = Vector

    def __init_by_shape(self, shape):
        if not isinstance(shape, tuple):
            raise TypeError("Shape must be a tuple")
        elif len(shape) != 2:
            raise ValueError("Shape must be a tuple of length 2")
        elif not all(isinstance(x, int) for x in shape):
            raise TypeError("Shape must contain only integers")
        elif not all(x > 0 for x in shape):
            raise ValueError("Shape must contain only positive integers")
        self.data = [[0.0 for _ in range(shape[1])]
                     for _ in range(shape[0])]
        self.shape = shape
        if self.shape[0] == 1 or self.shape[1] == 1:
            self.__class__ = Vector

    # add : only matrices/vectors of same dimensions.
    # __add__
    def __add__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError("Can only add a Matrix to a Matrix")
        elif self.shape != other.shape:
            raise ValueError("Can only add matrices of same shape")

        return Matrix([[a + b for a, b in zip(x, y)]
                       for x, y in zip(self.data, other.data)])

    # __radd__
    def __radd__(self, other):
        return self + other

    # sub : only matrices/vectors of same dimensions.
    # __sub__
    def __sub__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError("Can only subtract a Matrix from a Matrix")
        elif self.shape != other.shape:
            raise ValueError("Can only subtract matrices of same shape")

        return Matrix([[a - b for a, b in zip(x, y)]
                       for x, y in zip(self.data, other.data)])

    # __rsub__
    def __rsub__(self, other):
        return self - other

    # Multplies a matrix by a vector.
    def mul_by_vector(self, vector):
        if not isinstance(vector, Vector):
            raise TypeError("Can only multiply a Matrix by a Vector")
        elif self.shape[1] != vector.shape[0]:
            raise ValueError("Matrix and vector shapes do not match")
        data = []
        for i in range(self.shape[0]):
            val = 0
            for j in range(self.shape[1]):
                val += self.data[i][j] * vector.data[j][0]
            data.append([val])
        return Vector(data)

    # Multiplies a matrix by a scalar.
    def scale(self, factor):
        if not isinstance(factor, (int, float)):
            raise TypeError("Can only scale a Matrix by a scalar")
        return Matrix([[factor * x for x in row] for row in self.data])

    # Multiplies a matrix by a matrix.
    def mul_by_matrix(self, matrix):
        if not isinstance(matrix, Matrix):
            raise TypeError("Can only multiply a Matrix by a Matrix")
        elif self.shape[1] != matrix.shape[0]:
            raise ValueError("Matrix shapes do not match")
        data = []
        for i in range(self.shape[0]):
            row = []
            for j in range(matrix.shape[1]):
                val = 0
                for k in range(self.shape[1]):
                    val += self.data[i][k] * matrix.data[k][j]
                row.append(val)
            data.append(row)
        return Matrix(data)

    # mul : scalars, vectors and matrices,
    # can have errors with vectors and matrices,
    # returns a Vector if we perform Matrix * Vector mutliplication.
    # __mul__
    def __mul__(self, other):
        if isinstance(other, Vector):
            return self.mul_by_vector(other)
        elif isinstance(other, Matrix):
            return self.mul_by_matrix(other)
        elif isinstance(other, (int, float)):
            return self.scale(other)
        else:
            raise TypeError("Can only multiply a Matrix by a scalar, "
                            "a Vector or a Matrix")

    # __rmul__
    def __rmul__(self, other):
        return self * other

    # div : only scalars.
    # __truediv__
    def __truediv__(self, other):
        if not isinstance(other, (int, float)):
            raise TypeError("Can only divide a Matrix by a scalar")
        elif other == 0:
            raise ZeroDivisionError("Cannot divide a Matrix by 0")
        return self * (1 / other)

    # __rtruediv__
    def __rtruediv__(self, other):
        raise TypeError("Cannot divide a scalar by a Matrix")

    # __str__ : print the matrix in a nice way.
    def __str__(self):
        ret = type(self).__name__ + "(\n"
        for row in self.data:
            ret += " " + str(row) + "\n"
        ret += ")"

        return ret

    # __repr__
    # (More precise than __str__)
    def __repr__(self):
        if self.shape[0] == 1 or self.shape[1] == 1:
            return "Vector(" + str(self.data) + ")"
        else:
            return "Matrix(" + str(self.data) + ")"

    # Transpose the matrix
    def T(self):
        new = Matrix([[a for a in x] for x in zip(*self.data)])
        self.data = new.data
        self.shape = new.shape
        return self

    # ==
    def __eq__(self, other) -> bool:
        if isinstance(other, Matrix):
            return (self.data == other.data
                    and self.shape == other.shape)
        return False

    # !=
    def __ne__(self, other) -> bool:
        return not self == other


class Vector(Matrix):

    """
    A vector is a matrix with only one column.

    """

    def __init__(self,
                 data: 'list[list[float]]' = None,
                 shape: 'tuple[int, int]' = None):
        super().__init__(data=data, shape=shape)
        if self.shape[0] != 1 and self.shape[1] != 1:
            raise ValueError("A vector must have only one column")

    def dot(self, other):
        if not isinstance(other, Vector):
            raise TypeError("Can only dot a Vector with a Vector")
        elif self.shape != other.shape:
            raise ValueError("Can only dot vectors of same shape")
        return sum([self.data[i][0] * other.data[i][0]
                    for i in range(self.shape[0])])

    # Cross product
    def cross(self, other):
        if not isinstance(other, Vector):
            raise TypeError("Can only cross a Vector with a Vector")
        elif self.shape != other.shape:
            raise ValueError("Can only cross vectors of same shape")
        elif self.shape != (3, 1) and self.shape != (1, 3):
            raise ValueError(
                "Can only cross vectors of shape (3, 1) or (1, 3)")
        if self.shape == (1, 3):
            return Vector([
                self.data[0] * other.data[1] - self.data[1] * other.data[0],
                self.data[1] * other.data[2] - self.data[2] * other.data[1],
                self.data[2] * other.data[0] - self.data[0] * other.data[2]
            ])
        else:
            return Vector([
                [self.data[1][0] * other.data[2][0]
                 - self.data[2][0] * other.data[1][0]],
                [self.data[2][0] * other.data[0][0]
                 - self.data[0][0] * other.data[2][0]],
                [self.data[0][0] * other.data[1][0]
                 - self.data[1][0] * other.data[0][0]]
            ])


if __name__ == "__main__":

    # Subject tests part 1
    print("Subjet tests part 1")
    print("Matrix M1 :")
    m1 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    print(str(m1))
    print("M1 shape =", m1.shape)
    print("Transposition of M1 :")
    print(str(m1.T()))
    print("M1 shape =", m1.shape)

    # Subject tests part 2
    print("\nSubjet tests part 2")
    print("Matrix M2 :")
    m2 = Matrix([[0., 2., 4.], [1., 3., 5.]])
    print(str(m2))
    print("M2 shape =", m2.shape)
    print("Transposition of M2 :")
    print(str(m2.T()))
    print("M2 shape =", m2.shape)

    # Subject tests part 3
    print("\nSubjet tests part 3")
    m3 = Matrix([[0.0, 1.0, 2.0, 3.0],
                [0.0, 2.0, 4.0, 6.0]])

    m4 = Matrix([[0.0, 1.0],
                [2.0, 3.0],
                [4.0, 5.0],
                [6.0, 7.0]])
    m5 = m3 * m4
    print(str(m5))

    # Subject tests part 4
    print("\nSubjet tests part 4")
    print("Matrix M6 :")
    m6 = Matrix([[0.0, 1.0, 2.0],
                [0.0, 2.0, 4.0]])
    print(str(m6))
    print("M6 shape =", m6.shape)
    print("Vector V1 :")
    v1 = Vector([[1], [2], [3]])
    print(str(v1))
    print("V1 shape =", v1.shape)
    print("Multiplication of M6 and V1 :")
    print(str(m6 * v1))

    # Subject tests part 5
    print("\nSubjet tests part 5")
    print("Vector V2 :")
    v2 = Vector([[1], [2], [3]])
    print(str(v2))
    print("V2 shape =", v2.shape)
    print("Vector V3 :")
    v3 = Vector([[2], [4], [8]])
    print(str(v3))
    print("V3 shape =", v3.shape)
    print("Addition of V2 and V3 :")
    print(str(v2 + v3))
