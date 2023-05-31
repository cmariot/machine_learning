from matrix import Matrix  # , Vector
import unittest


class TestMatrix(unittest.TestCase):

    def test_init_with_data(self):
        # Test initializing matrix with data
        data = [[1, 2], [3, 4]]
        mx = Matrix(data=data)
        self.assertEqual(mx.data, data)
        self.assertEqual(mx.shape, (2, 2))

        # Test initializing matrix with invalid data
        with self.assertRaises(TypeError):
            Matrix(data="invalid")

        with self.assertRaises(ValueError):
            Matrix(data=[[1, 2], [3]])

        with self.assertRaises(TypeError):
            Matrix(data=[[1, 2], [3, "invalid"]])

    def test_init_with_shape(self):
        # Test initializing matrix with shape
        shape = (2, 2)
        mx = Matrix(shape=shape)
        self.assertEqual(mx.data, [[0.0, 0.0], [0.0, 0.0]])
        self.assertEqual(mx.shape, shape)

        # Test initializing matrix with invalid shape
        with self.assertRaises(TypeError):
            Matrix(shape="invalid")

        with self.assertRaises(TypeError):
            Matrix(shape=(2, "invalid"))

        with self.assertRaises(ValueError):
            Matrix(shape=(-1, 2))

    def test_add(self):
        # Test adding matrices of same dimensions
        matrix1 = Matrix(data=[[1, 2], [3, 4]])
        matrix2 = Matrix(data=[[5, 6], [7, 8]])
        result = matrix1 + matrix2
        self.assertEqual(result.data, [[6, 8], [10, 12]])
        self.assertEqual(result.shape, (2, 2))

        # Test adding matrices of different dimensions
        matrix3 = Matrix(data=[[1, 2], [3, 4]])
        matrix4 = Matrix(data=[[5, 6, 7], [8, 9, 10]])
        with self.assertRaises(ValueError):
            matrix3 + matrix4

    def test_add_with_scalar(self):
        # Test adding a scalar to a matrix
        matrix1 = Matrix(data=[[1, 2], [3, 4]])
        with self.assertRaises(TypeError):
            matrix1 + 2

        # Test adding a scalar to an invalid type
        with self.assertRaises(TypeError):
            matrix1 + "invalid"

    def test_radd(self):
        # Test right addition with a matrix
        matrix1 = Matrix(data=[[1, 2], [3, 4]])
        matrix2 = Matrix(data=[[2, 3], [4, 5]])
        result = matrix2 + matrix1
        self.assertEqual(result.data, [[3, 5], [7, 9]])
        self.assertEqual(result.shape, (2, 2))

        # Test right addition with a scalar
        with self.assertRaises(TypeError):
            2 + matrix1

        # Test right addition with an invalid type
        with self.assertRaises(TypeError):
            "invalid" + matrix1

    def test_sub(self):
        # Test subtracting matrices of same dimensions
        matrix1 = Matrix(data=[[1, 2], [3, 4]])
        matrix2 = Matrix(data=[[5, 6], [7, 8]])
        result = matrix1 - matrix2
        self.assertEqual(result.data, [[-4, -4], [-4, -4]])
        self.assertEqual(result.shape, (2, 2))

        # Test subtracting matrices of different dimensions
        matrix3 = Matrix(data=[[1, 2], [3, 4]])
        matrix4 = Matrix(data=[[5, 6, 7], [8, 9, 10]])
        with self.assertRaises(ValueError):
            matrix3 - matrix4

        # Test subtracting an invalid type
        with self.assertRaises(TypeError):
            matrix1 - "invalid"

    def test_rsub(self):
        # Test right subtraction with a matrix
        matrix1 = Matrix(data=[[1, 2], [3, 4]])
        matrix2 = Matrix(data=[[2, 3], [4, 5]])
        result = matrix2 - matrix1
        self.assertEqual(result.data, [[1, 1], [1, 1]])
        self.assertEqual(result.shape, (2, 2))

        # Test right subtraction with a scalar
        with self.assertRaises(TypeError):
            2 - matrix1

        # Test right subtraction with an invalid type
        with self.assertRaises(TypeError):
            "invalid" - matrix1

    def test_mul(self):

        matrix1 = Matrix([[1, 2], [3, 4]])
        matrix2 = Matrix([[5, 6], [7, 8]])
        scalar1 = 2

        # Test matrix * matrix multiplication
        result = matrix1 * matrix2
        expected = Matrix([[19, 22], [43, 50]])
        self.assertEqual(result, expected)

        # Test matrix * scalar multiplication
        result = matrix1 * scalar1
        expected = Matrix([[2, 4], [6, 8]])
        self.assertEqual(result, expected)

        # Test invalid input type
        with self.assertRaises(TypeError):
            matrix1 * "string"

        # Test invalid matrix dimensions
        print(matrix1 * matrix1.T())

    def test_rmul(self):

        matrix1 = Matrix([[1, 2], [3, 4]])
        scalar1 = 2

        # Test scalar * matrix multiplication
        result = scalar1 * matrix1
        expected = Matrix([[2, 4], [6, 8]])
        self.assertEqual(result, expected)

        # Test invalid input type
        with self.assertRaises(TypeError):
            "string" * matrix1

    def test_div(self):
        matrix1 = Matrix(data=[[1, 2], [3, 4]])
        result = matrix1 / 2
        expected = Matrix([[0.5, 1.0], [1.5, 2.0]])
        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result.data, expected.data)

        with self.assertRaises(TypeError):
            matrix2 = Matrix(data=[[1.0, 2.0], [3, 4]])
            matrix1 / matrix2

        with self.assertRaises(TypeError):
            matrix1 / "invalid"

        with self.assertRaises(ZeroDivisionError):
            matrix1 / 0

    def test_rdiv(self):
        matrix1 = Matrix(data=[[1, 2], [3, 4]])
        result = 2 / matrix1
        expected = Matrix([[2.0, 1.0], [2.0 / 3, 0.5]])
        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result.data, expected.data)

        with self.assertRaises(TypeError):
            "invalid" / matrix1

        with self.assertRaises(ZeroDivisionError):
            matrix2 = Matrix([[0, 1.0], [2.0 / 3, 0.5]])
            2 / matrix2

    def test_T(self):

        matrix1 = Matrix([[1, 2], [3, 4]])
        matrix1.T()
        expected = Matrix([[1.0, 3.0], [2.0, 4.0]])
        self.assertEqual(matrix1, expected)

        matrix2 = Matrix([[5, 6, 7], [8, 9, 10]])
        matrix2.T()
        expected = Matrix([[5, 8], [6, 9], [7, 10]])
        self.assertEqual(matrix2, expected)

        matrix3 = Matrix([[1, 2], [3, 4], [5, 6]])
        matrix3.T()
        expected = Matrix([[1, 3, 5], [2, 4, 6]])
        self.assertEqual(matrix3, expected)

        matrix4 = Matrix([[1]])
        matrix4.T()
        expected = Matrix([[1]])
        self.assertEqual(matrix4, expected)


if __name__ == '__main__':
    unittest.main()
