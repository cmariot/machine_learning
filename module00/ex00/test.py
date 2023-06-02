from matrix import Matrix, Vector
import unittest


class TestVector(unittest.TestCase):

    def test_init_by_data(self):
        data = [[1], [2], [3]]
        vec = Vector(data)
        self.assertEqual(vec.data, [[1.0], [2.0], [3.0]])
        self.assertEqual(vec.shape, (3, 1))
        data = [[1, 2, 3]]
        vec = Vector(data)
        self.assertEqual(vec.data, [[1., 2., 3.]])
        self.assertEqual(vec.shape, (1, 3))

    def test_init_by_shape(self):
        shape = (3, 1)
        vec = Vector(shape=shape)
        self.assertEqual(vec.data, [[0.0], [0.0], [0.0]])
        self.assertEqual(vec.shape, shape)

        shape = (1, 3)
        vec = Vector(shape=shape)
        self.assertEqual(vec.data, [[0., 0., 0.]])
        self.assertEqual(vec.shape, shape)

    def test_init_with_invalid_data(self):
        with self.assertRaises(TypeError):
            Vector()
        with self.assertRaises(TypeError):
            Vector(foo="bar")
        with self.assertRaises(ValueError):
            Vector([])
        with self.assertRaises(ValueError):
            Vector([[]])
        with self.assertRaises(TypeError):
            Vector([['a']])
        with self.assertRaises(ValueError):
            Vector([[1, 2], [3, 4]])
        with self.assertRaises(TypeError):
            Vector([1, 2, 3])
        with self.assertRaises(TypeError):
            Vector([[1, 2, 3]], shape=(3, 1))

    def test_init_with_invalid_shape(self):
        with self.assertRaises(TypeError):
            Vector(shape=1)
        with self.assertRaises(TypeError):
            Vector(shape=('1', 3))
        with self.assertRaises(TypeError):
            Vector(shape=(3, '1'))
        with self.assertRaises(TypeError):
            Vector(shape=3)
        with self.assertRaises(ValueError):
            Vector(shape=(3, 1, 1))
        with self.assertRaises(ValueError):
            Vector(shape=(3, -1))
        with self.assertRaises(ValueError):
            Vector(shape=(0, 0))
        with self.assertRaises(ValueError):
            Vector(shape=(3, 0))

    def test_add(self):
        vec1 = Vector([[1, 2, 3]])
        vec2 = Vector([[4, 5, 6]])
        vec3 = vec1 + vec2
        self.assertEqual(vec3.data, [[5.0, 7.0, 9.0]])
        self.assertEqual(vec3.shape, (1, 3))

        vec4 = Vector([[1], [2], [3]])
        vec5 = Vector([[4], [5], [6]])
        vec6 = vec4 + vec5
        self.assertEqual(vec6.data, [[5.0], [7.0], [9.0]])

        with self.assertRaises(ValueError):
            vec1 + vec4

        with self.assertRaises(TypeError):
            vec4 + 2

    def test_sub(self):
        vec1 = Vector([[1, 2, 3]])
        vec2 = Vector([[4, 5, 6]])
        vec3 = vec1 - vec2
        self.assertEqual(vec3.data, [[-3.0, -3.0, -3.0]])
        self.assertEqual(vec3.shape, (1, 3))

        vec4 = Vector([[1], [2], [3]])
        vec5 = Vector([[4], [5], [6]])
        vec6 = vec4 - vec5
        self.assertEqual(vec6.data, [[-3.0], [-3.0], [-3.0]])

        with self.assertRaises(ValueError):
            vec1 - vec4

        with self.assertRaises(TypeError):
            vec4 - 2

    def test_scale(self):
        mat1 = Matrix([[1, 2], [3, 4]])
        mat2 = mat1 * 2
        self.assertEqual(mat2.data, [[2.0, 4.0], [6.0, 8.0]])
        self.assertEqual(mat2.shape, (2, 2))

    def test_scale_with_invalid_factor(self):
        mat1 = Matrix([[1, 2], [3, 4]])
        with self.assertRaises(TypeError):
            mat1.scale('2')


class TestMatrix(unittest.TestCase):

    def test_init_by_data(self):
        data = [[1], [2], [3]]
        vec = Matrix(data)
        self.assertEqual(vec.data, [[1.0], [2.0], [3.0]])
        self.assertEqual(vec.shape, (3, 1))
        self.assertEqual(type(vec), Vector)

        data = [[1, 2, 3]]
        vec = Matrix(data)
        self.assertEqual(vec.data, [[1., 2., 3.]])
        self.assertEqual(vec.shape, (1, 3))
        self.assertEqual(type(vec), Vector)

        data = [[1, 2, 3], [4, 5, 6]]
        mat = Matrix(data)
        self.assertEqual(mat.data, [[1., 2., 3.], [4., 5., 6.]])
        self.assertEqual(mat.shape, (2, 3))
        self.assertEqual(type(mat), Matrix)

    def test_init_by_shape(self):
        shape = (3, 1)
        vec = Matrix(shape=shape)
        self.assertEqual(vec.data, [[0.0], [0.0], [0.0]])
        self.assertEqual(vec.shape, shape)
        self.assertEqual(type(vec), Vector)

        shape = (1, 3)
        vec = Matrix(shape=shape)
        self.assertEqual(vec.data, [[0., 0., 0.]])
        self.assertEqual(vec.shape, shape)
        self.assertEqual(type(vec), Vector)

        shape = (2, 3)
        mat = Matrix(shape=shape)
        self.assertEqual(mat.data, [[0., 0., 0.], [0., 0., 0.]])
        self.assertEqual(mat.shape, shape)
        self.assertEqual(type(mat), Matrix)

    def test_init_with_invalid_data(self):
        with self.assertRaises(TypeError):
            Matrix()
        with self.assertRaises(TypeError):
            Matrix(foo="bar")
        with self.assertRaises(ValueError):
            Matrix([])
        with self.assertRaises(ValueError):
            Matrix([[]])
        with self.assertRaises(TypeError):
            Matrix([['a']])
        with self.assertRaises(TypeError):
            Matrix([1, 2, 3])
        with self.assertRaises(TypeError):
            Matrix([[1, 2, 3]], shape=(3, 1))
        with self.assertRaises(ValueError):
            Matrix([[1, 2], [3, 4, 5]])
        with self.assertRaises(TypeError):
            Matrix([[1, 2], [3, '4']])
        with self.assertRaises(TypeError):
            Matrix([[1, 2], [3, [4]]])

    def test_init_with_invalid_shape(self):
        with self.assertRaises(TypeError):
            Matrix(shape=1)
        with self.assertRaises(TypeError):
            Matrix(shape=('1', 3))
        with self.assertRaises(TypeError):
            Matrix(shape=(3, '1'))
        with self.assertRaises(TypeError):
            Matrix(shape=3)
        with self.assertRaises(ValueError):
            Matrix(shape=(3, 1, 1))
        with self.assertRaises(ValueError):
            Matrix(shape=(3, -1))
        with self.assertRaises(ValueError):
            Matrix(shape=(0, 0))
        with self.assertRaises(ValueError):
            Matrix(shape=(3, 0))

    def test_add(self):
        mat1 = Matrix([[1, 2], [3, 4]])
        mat2 = Matrix([[5, 6], [7, 8]])
        mat3 = mat1 + mat2
        self.assertEqual(mat3.data, [[6.0, 8.0], [10.0, 12.0]])
        self.assertEqual(mat3.shape, (2, 2))

    def test_sub(self):
        mat1 = Matrix([[1, 2], [3, 4]])
        mat2 = Matrix([[5, 6], [7, 8]])
        mat3 = mat1 - mat2
        self.assertEqual(mat3.data, [[-4.0, -4.0], [-4.0, -4.0]])
        self.assertEqual(mat3.shape, (2, 2))

    def test_scale(self):
        vec1 = Vector([[1, 2, 3]])
        vec2 = vec1 * 2.
        self.assertEqual(vec2.data, [[2.0, 4.0, 6.0]])
        self.assertEqual(vec2.shape, (1, 3))

    def test_scale_with_invalid_factor(self):
        vec1 = Vector([[1, 2, 3]])
        with self.assertRaises(TypeError):
            vec1.scale('2')

    def test_matrix_vector_multiplication(self):
        m1 = Matrix([[0.0, 1.0, 2.0],
                     [0.0, 2.0, 4.0]])
        v1 = Vector([[1], [2], [3]])
        expected = [[8.0], [16.0]]
        actual = m1 * v1
        self.assertEqual(actual.data, expected)
        self.assertEqual(actual.shape, (2, 1))

    def test_matrix_matrix_multiplication(self):
        m1 = Matrix([[0.0, 1.0, 2.0, 3.0],
                     [0.0, 2.0, 4.0, 6.0]])
        m2 = Matrix([[0.0, 1.0],
                     [2.0, 3.0],
                     [4.0, 5.0],
                     [6.0, 7.0]])
        mat3 = m1 * m2
        expected = Matrix([[28., 34.], [56., 68.]])
        self.assertEqual(mat3.data, expected.data)
        self.assertEqual(mat3.shape, (2, 2))

        mat4 = Matrix([[1, 2, 3], [4, 5, 6]])
        mat5 = Matrix([[1, 2], [3, 4], [5, 6]])
        mat6 = mat4 * mat5
        self.assertEqual(mat6.data, [[22.0, 28.0], [49.0, 64.0]])
        self.assertEqual(mat6.shape, (2, 2))


if __name__ == '__main__':
    unittest.main()
