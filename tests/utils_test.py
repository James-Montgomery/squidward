import unittest
import warnings
import numpy as np
import numpy.testing as npt

from squidward import utils
from squidward.utils import deprecated

# useful for debugging
np.set_printoptions(suppress=True)


class UtilitiesTestCase(unittest.TestCase):
    """Class for utilities tests."""

    # ---------------------------------------------------------------------------------------------------------------------
    # Array Checks
    # ---------------------------------------------------------------------------------------------------------------------

    def test_1(self):
        """
        Exactly 1D
        Test that at 1d always returns a 1d array.
        """
        x = true = np.ones((10))
        output = utils.exactly_1d(x)
        npt.assert_almost_equal(output, x, decimal=10)

        x = np.ones((10,1))
        output = utils.exactly_1d(x)
        npt.assert_almost_equal(output, true, decimal=10)

        x = np.ones((1,10))
        output = utils.exactly_1d(x)
        npt.assert_almost_equal(output, true, decimal=10)

        x = np.ones((2,10))
        with self.assertRaises(Exception) as context:
            utils.exactly_1d(x)
        self.assertTrue('Not appropriate input shape.' in str(context.exception))

        x = np.ones((2,10,1))
        with self.assertRaises(Exception) as context:
            utils.exactly_1d(x)
        self.assertTrue('Not appropriate input shape.' in str(context.exception))

    def test_2(self):
        """
        Exactly 2D
        Test that at least 2d always returns a >= 2d array.
        """
        true = np.ones(10).reshape(-1,1)

        x = np.ones(10).reshape(-1,1)
        output = utils.exactly_2d(x)
        npt.assert_almost_equal(output, true, decimal=10)

        x = np.ones(10).reshape(1,-1)
        output = utils.exactly_2d(x)
        npt.assert_almost_equal(output, true, decimal=10)

        x = np.ones(10)
        output = utils.exactly_2d(x)
        npt.assert_almost_equal(output, true, decimal=10)

        true = np.ones((10,10))

        x = np.ones((10,10,1))
        output = utils.exactly_2d(x)
        npt.assert_almost_equal(output, true, decimal=10)

        x = np.ones((1,10,10))
        output = utils.exactly_2d(x)
        npt.assert_almost_equal(output, true, decimal=10)

        x = np.ones((10,1,10))
        with self.assertRaises(Exception) as context:
            utils.exactly_2d(x)
        self.assertTrue('Not appropriate input shape.' in str(context.exception))

        x = np.ones((10,1,10,1))
        with self.assertRaises(Exception) as context:
            utils.exactly_2d(x)
        self.assertTrue('Not appropriate input shape.' in str(context.exception))

    # ---------------------------------------------------------------------------------------------------------------------
    # Inversions
    # ---------------------------------------------------------------------------------------------------------------------

    def test_3(self):
        """
        Is Intertible True
        Test that non-singular matricies return true.
        """
        arr = np.random.rand(10, 10)
        arr = arr.dot(arr.T)

        output = utils.is_invertible(arr, 'condition')
        assert output

        output = utils.is_invertible(arr, 'rank')
        assert output

        # cramer's rule method fails here due to
        # floating point errors in np.linalg.det
        # LU decomposition approximation of determinant
        output = utils.is_invertible(arr, 'cramer')
        assert ~output

    def test_4(self):
        """
        Is Invertible False
        Test that singular matricies return false.
        """
        arr = np.random.rand(10, 10)
        arr[-1] = arr[0] + arr[1]

        output = utils.is_invertible(arr, 'condition')
        assert ~output

        output = utils.is_invertible(arr, 'rank')
        assert ~output

        output = utils.is_invertible(arr, 'cramer')
        assert ~output

    def test_5(self):
        """
        Check Valid Covariance
        Test that the function that validates covariance matricies works.
        """
        x = np.array([[1, 1, 1],[1, 0, 1],[1, 1, 0]])
        output = output = utils.check_valid_cov(x)
        assert output is None

        x = np.array([[-1, 1, 1],[1, 0, 1],[1, 1, 0]])
        with self.assertRaises(Exception) as context:
            utils.check_valid_cov(x)
        self.assertTrue('Negative values in diagonal of covariance matrix.\nLikely cause is kernel '
                        'inversion instability.\nCheck kernel variance.' in str(context.exception))

        # pass a singular matrix
        arr = np.random.rand(10, 10)
        arr[-1] = arr[0] + arr[1]
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            utils.check_valid_cov(arr)
            assert "Cov has high condition. Inverting matrix may result in errors." in str(w[-1].message)

        # TODO: check show_warnings argument actually silences warnings

    def test_6(self):
        """
        Invert
        Test that inversion methods work on a typical matrix input with a
        reasonable condition.
        """

        # create a test input that is not singular
        arr = np.array([[0.08647087, 0.44631909, 0.20543369, 0.80556576, 0.484415  ],
                        [0.83409753, 0.7406405 , 0.72326909, 0.59616491, 0.86701306],
                        [0.83761527, 0.49645837, 0.64037925, 0.95100387, 0.13899134],
                        [0.97684547, 0.30623548, 0.95194714, 0.28353989, 0.831871  ],
                        [0.45327912, 0.74906165, 0.94224464, 0.30019356, 0.56802402]])
        arr = arr.dot(arr.T)

        true = np.array([[ 6.39675434, -7.5605537 , -1.01890231,  4.9418642 ,  0.76873378],
                         [-7.5605537 , 15.51247025, -0.63188021, -8.45294531, -3.96604294],
                         [-1.01890231, -0.63188021,  1.95649949, -0.52470477,  0.14555444],
                         [ 4.9418642 , -8.45294531, -0.52470477,  6.88240616,  0.11092939],
                         [ 0.76873378, -3.96604294,  0.14555444,  0.11092939,  4.23098611]])

        inv = utils.Invert("inv")
        output = inv(arr)
        npt.assert_almost_equal(output, true, decimal=7)

        inv = utils.Invert("pinv")
        output = inv(arr)
        npt.assert_almost_equal(output, true, decimal=7)

        inv = utils.Invert("solve")
        output = inv(arr)
        npt.assert_almost_equal(output, true, decimal=7)

        inv = utils.Invert("cholesky")
        output = inv(arr)
        npt.assert_almost_equal(output, true, decimal=7)

        inv = utils.Invert("svd")
        output = inv(arr)
        npt.assert_almost_equal(output, true, decimal=7)

        inv = utils.Invert("lu")
        output = inv(arr)
        npt.assert_almost_equal(output, true, decimal=7)

        # pass an invlaid inversion method
        with self.assertRaises(Exception) as context:
            utils.Invert("fake")
        self.assertTrue('Invalid inversion method argument.' in str(context.exception))

        # pass a singular matrix and catch warning

        arr = np.random.rand(10, 10)
        arr[-1] = arr[0] + arr[1]

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            inv = utils.Invert()
            inv(arr)
            # Verify some things
            assert len(w) == 1
            assert "Matrix has high condition." in str(w[-1].message)

    # ---------------------------------------------------------------------------------------------------------------------
    # Pre-processing
    # ---------------------------------------------------------------------------------------------------------------------

    def test_7(self):
        """
        Onehot
        Test that one hot returns the appropriate one hot array.
        """
        y = np.array([0,1,0,2,1,0,1,2,0,1])
        true = np.array([[1., 0., 0.],
                         [0., 1., 0.],
                         [1., 0., 0.],
                         [0., 0., 1.],
                         [0., 1., 0.],
                         [1., 0., 0.],
                         [0., 1., 0.],
                         [0., 0., 1.],
                         [1., 0., 0.],
                         [0., 1., 0.]])

        output = utils.onehot(y,3)
        npt.assert_almost_equal(output, true, decimal=10)

        output = utils.onehot(y)
        npt.assert_almost_equal(output, true, decimal=10)

        with self.assertRaises(Exception) as context:
            utils.onehot(y,4)
        self.assertTrue('Number of unique values does not match num_classes argument.' in str(context.exception))

        with self.assertRaises(Exception) as context:
            utils.onehot(y,2)
        self.assertTrue('Number of unique values does not match num_classes argument.' in str(context.exception))

    def test_8(self):
        """
        Reversehot
        Test that reverse hot appropriately reverses one hot arrays. Should do
        the exact opposite of one hot fucntion.
        """
        y = np.array([[1., 0., 0.],
                      [1., 0., 0.],
                      [1., 0., 0.],
                      [1., 0., 0.],
                      [0., 1., 0.],
                      [1., 0., 0.],
                      [0., 0., 1.],
                      [0., 0., 1.],
                      [0., 0., 1.],
                      [1., 0., 0.]])

        true = np.array([0.0,0.0,0.0,0.0,1.0,0.0,2.0,2.0,2.0,0.0])

        output = utils.reversehot(y)
        npt.assert_almost_equal(output, true, decimal=10)

        output = utils.reversehot(true)
        npt.assert_almost_equal(output, true, decimal=10)

        output = utils.reversehot(true.T)
        npt.assert_almost_equal(output, true, decimal=10)

        y = true.reshape(-1,1)
        output = utils.reversehot(y)
        npt.assert_almost_equal(output, true, decimal=10)

        y = true.reshape(1,-1)
        output = utils.reversehot(y)
        npt.assert_almost_equal(output, true, decimal=10)

    # ---------------------------------------------------------------------------------------------------------------------
    # Classification Specific
    # ---------------------------------------------------------------------------------------------------------------------

    def test_9(self):
        """
        Sigmoid
        Test sigmoid functions works.
        """

        x = 0.458
        true = 0.61253961344091512
        output = utils.sigmoid(x)
        npt.assert_almost_equal(output, true, decimal=10)

        x = np.array([[-8, 0, 6],[8, 3, 1],[10, -300, 11]])
        true = np.array([[0.0003353501, 0.5         , 0.9975273768],
                         [0.9996646499, 0.9525741268, 0.7310585786],
                         [0.9999546021, 0.          , 0.9999832986]])

        output = utils.sigmoid(x)
        npt.assert_almost_equal(output, true, decimal=10)

        output = utils.sigmoid(-12345)
        true = 0.0
        npt.assert_almost_equal(output, true, decimal=10)

        output = utils.sigmoid(12345)
        true = 1.0
        npt.assert_almost_equal(output, true, decimal=10)

    def test_10(self):
        """
        Softmax
        Test softmax function works.
        """
        x = np.array([[-8, 0, 6],[8, 3, 1],[10, -300, 11]])
        true = np.array([[2.2388575697e-004, 3.3380896059e-001, 6.6596715365e-001],
                         [3.7255082739e-001, 3.5500132884e-001, 2.7244784376e-001],
                         [4.9999282567e-001, 2.5741800386e-131, 5.0000717433e-001]])

        output = utils.softmax(x)
        npt.assert_almost_equal(output, true, decimal=10)

    # ---------------------------------------------------------------------------------------------------------------------
    # Miscellaneous
    # ---------------------------------------------------------------------------------------------------------------------

    def test_11(self):
        """
        Deprecated
        Ensure that the deprecated warning actually returns the right warning
        to the user.
        """
        @deprecated
        def f():
            pass

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            f()
            # Verify some things
            assert len(w) == 1
            assert issubclass(w[-1].category, DeprecationWarning)
            assert "deprecated" in str(w[-1].message)


if __name__ == '__main__':
    unittest.main()
