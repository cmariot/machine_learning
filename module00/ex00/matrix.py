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
        elif data is not None:
            self.__init_by_data(data)
        elif shape is not None:
            self.__init_by_shape(shape)

    def __init_by_data(self, data):
        if (not isinstance(data, list)
                or not all(isinstance(x, list) for x in data)):
            raise TypeError("Data must be a list of lists")
        elif not all(len(x) == len(data[0]) for x in data):
            raise ValueError(
                "Data must be a matrix, all rows must have the same length")
        elif not all(
                isinstance(x, (int, float)) for row in data for x in row):
            raise TypeError("Data must contain only integers or floats")
        self.data = [[float(x) for x in row] for row in data]
        self.shape = (len(data), len(data[0]))

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

    # add : only matrices of same dimensions.
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

    # sub : only matrices of same dimensions.
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
        if not isinstance(other, (int, float)):
            raise TypeError("Can only divide a scalar by a Matrix")
        elif any(0 in row for row in self.data):
            raise ZeroDivisionError(
                "Cannot divide by a Matrix with a 0 element")
        
        # Check the return value, not sure if it's correct.

        return Matrix([[other / a for a in row] for row in self.data])

    # mul : scalars, vectors and matrices,
    # can have errors with vectors and matrices,
    # returns a Vector if we perform Matrix * Vector mutliplication.
    # __mul__
    def __mul__(self, other):
        if isinstance(other, Matrix):
            if self.shape[1] != other.shape[0]:
                raise ValueError("Can only multiply matrices" +
                                 " if the number of columns of the first" +
                                 " matrix is equal to the number of rows" +
                                 " of the second matrix")
            return Matrix([[sum(a * b for a, b in zip(x, y))
                            for y in zip(*other.data)]
                           for x in self.data])
        elif isinstance(other, Vector):
            if self.shape[1] != other.shape[0]:
                raise ValueError("Can only multiply a matrix by a vector" +
                                 " if the number of columns of the matrix" +
                                 " is equal to the number of rows of" +
                                 " the vector")
            return Vector([sum(a * b for a, b in zip(x, other.data))
                           for x in self.data])
        elif isinstance(other, (int, float)):
            return Matrix([[other * a for a in row] for row in self.data])
        else:
            raise TypeError("Can only multiply a Matrix by a scalar," +
                            " a Vector or a Matrix")

    # __rmul__
    def __rmul__(self, other):
        return self * other

    # __str__ : print the matrix in a nice way.
    def __str__(self):
        max_len = max(len(str(x)) for row in self.data for x in row) + 2
        string = ""
        for row in self.data:
            string += "[ "
            for x in row:
                string += str(x).rjust(max_len)
            string += "]\n"
        return string[:-1]

    # __repr__
    def __repr__(self):
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

    def __init__(self, data):
        if not isinstance(data, list):
            raise TypeError("Data must be a list")
        elif not all(isinstance(x, list) for x in data):
            raise TypeError("Data must be a list of lists")
        elif not all(isinstance(x, (int, float)) for row in data for x in row):
            raise TypeError("Data must be a list of lists of numbers")

        if len(data) == 1:
            super().__init__(data, (1, len(data[0])))
        else:
            if not all(len(row) == 1 for row in data):
                raise ValueError("Data must be a list of lists of size (n, 1)")
            super().__init__(data, (len(data), 1))

    # add : only vectors of same dimensions.
    # __add__
    def __add__(self, other):
        if not isinstance(other, Vector):
            raise TypeError("Can only add a Vector to a Vector")
        elif self.shape != other.shape:
            raise ValueError("Can only add vectors of same shape")

        return Vector([[a + b] for a, b in zip(self.data, other.data)])

    # __radd__
    def __radd__(self, other):
        return self + other

    # sub : only vectors of same dimensions.
    # __sub__
    def __sub__(self, other):
        if not isinstance(other, Vector):
            raise TypeError("Can only subtract a Vector from a Vector")
        elif self.shape != other.shape:
            raise ValueError("Can only subtract vectors of same shape")

        return Vector([[a - b] for a, b in zip(self.data, other.data)])

    # __rsub__
    def __rsub__(self, other):
        return self - other

    # div : only scalars.
    # __truediv__
    def __truediv__(self, other):
        if not isinstance(other, (int, float)):
            raise TypeError("Can only divide a Vector by a scalar")
        elif other == 0:
            raise ZeroDivisionError("Cannot divide a Vector by 0")
        return self * (1 / other)

    # __rtruediv__
    def __rtruediv__(self, other):
        if not isinstance(other, (int, float)):
            raise TypeError("Can only divide a scalar by a Vector")
        elif any(0 in row for row in self.data):
            raise ZeroDivisionError(
                "Cannot divide by a Vector with a 0 element")
        return Vector([[other / a] for a in self.data])

    # mul : scalars, vectors and matrices,
    # can have errors with vectors and matrices,
    # returns a Vector if we perform Matrix * Vector mutliplication.
    # __mul__
    def __mul__(self, other):
        if isinstance(other, Matrix):
            if self.shape[0] != other.shape[1]:
                raise ValueError("Can only multiply a vector by a matrix" +
                                 " if the number of rows of the vector" +
                                 " is equal to the number of columns of" +
                                 " the matrix")
            return Vector([[sum(a * b for a, b in zip(x, other.data))]
                           for x in self.data])
        elif isinstance(other, Vector):
            if self.shape != other.shape:
                raise ValueError("Can only multiply vectors of same shape")
            return sum(a * b for a, b in zip(self.data, other.data))
        elif isinstance(other, (int, float)):
            return Vector([[other * a] for a in self.data])
        else:
            raise TypeError("Can only multiply a Vector by a scalar," +
                            " a Vector or a Matrix")

    # __rmul__
    def __rmul__(self, other):
        return self * other

    # __str__
    def __str__(self):
        max_len = max(len(str(x)) for x in self.data) + 2
        string = "[ "
        for x in self.data:
            string += str(x).rjust(max_len)
        string += "]"
        return string

    # __repr__
    def __repr__(self):
        return "Vector(" + str(self.data) + ")"

    # Dot product
    def dot(self, other):
        return self * other

    # Cross product
    def cross(self, other):
        if not isinstance(other, Vector):
            raise TypeError("Can only cross a Vector with a Vector")
        elif self.shape != other.shape:
            raise ValueError("Can only cross vectors of same shape")
        elif self.shape != (3, 1):
            raise ValueError("Can only cross vectors of shape (3, 1)")
        return Vector([[self.data[1][0] * other.data[2][0] -
                        self.data[2][0] * other.data[1][0]],
                       [self.data[2][0] * other.data[0][0] -
                        self.data[0][0] * other.data[2][0]],
                       [self.data[0][0] * other.data[1][0] -
                        self.data[1][0] * other.data[0][0]]])
