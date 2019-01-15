import unittest
import warnings
import numpy as np
import numpy.testing as npt
from squidward import utils
from squidward.utils import deprecated
np.random.seed(0)

# useful for debugging
np.set_printoptions(suppress=True)

class utilsTestCase(unittest.TestCase):
    """
    Class for utilities tests.
    """
    def test_sigmoid(self):
        """
        Test sigmoid functions works.
        """
        x = np.array([[-8,0,6],[8,3,1],[10,-300,11]])
        true = np.array([[3.3535013047e-004, 5.0000000000e-001, 9.9752737684e-001],
                         [9.9966464987e-001, 9.5257412682e-001, 7.3105857863e-001],
                         [9.9995460213e-001, 5.1482002224e-131, 9.9998329858e-001]])

        output = utils.sigmoid(x)
        npt.assert_almost_equal(output, true, decimal=10)

        output = utils.sigmoid(-12345, True)
        true = 0.5
        npt.assert_almost_equal(output, true, decimal=10)

        with self.assertRaises(Exception) as context:
           utils.sigmoid(-12345)
        self.assertTrue('overflow encountered in exp' in str(context.exception))

    def test_softmax(self):
        """
        Test softmax function works.
        """
        x = np.array([[-8,0,6],[8,3,1],[10,-300,11]])
        x = utils.sigmoid(x)
        true = np.array([[2.2388575697e-004, 3.3380896059e-001, 6.6596715365e-001],
                         [3.7255082739e-001, 3.5500132884e-001, 2.7244784376e-001],
                         [4.9999282567e-001, 2.5741800386e-131, 5.0000717433e-001]])

        output = utils.softmax(x)
        npt.assert_almost_equal(output, true, decimal=10)

    def test_is_invertible_true(self):
        """
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

    def test_is_invertible_false(self):
        """
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

    def test_check_valid_cov(self):
        """
        Test that the function that validates covariance matricies works.
        """
        x = np.array([[1,1,1],[1,0,1],[1,1,0]])
        output = output = utils.check_valid_cov(x)
        assert output is None

        x = np.array([[-1,1,1],[1,0,1],[1,1,0]])
        with self.assertRaises(Exception) as context:
            utils.check_valid_cov(x)
        self.assertTrue('Negative values in diagonal of covariance matrix.\nLikely cause is kernel inversion instability.\nCheck kernel variance.' in str(context.exception))

    def test_atleast_2d(self):
        """
        Test that at least 2d always returns a >= 2d array.
        """
        true = np.ones(10).reshape(-1,1)

        x = np.ones(10)
        output = utils.atleast_2d(x)
        npt.assert_almost_equal(output, true, decimal=10)

        x = np.ones(10).reshape(1,-1)
        output = utils.atleast_2d(x)
        npt.assert_almost_equal(output, true, decimal=10)

        x = np.ones((10,10,2))
        output = utils.atleast_2d(x)
        npt.assert_almost_equal(output, x, decimal=10)

    def test_atmost_1d(self):
        """
        Test that at 1d always returns a 1d array.
        """
        x = true = np.ones((10))
        output = utils.atmost_1d(x)
        npt.assert_almost_equal(output, x, decimal=10)

        x = np.ones((10,1))
        output = utils.atmost_1d(x)
        npt.assert_almost_equal(output, true, decimal=10)

        x = np.ones((1,10))
        output = utils.atmost_1d(x)
        npt.assert_almost_equal(output, true, decimal=10)

        x = np.ones((2,10))
        with self.assertRaises(Exception) as context:
            output = utils.atmost_1d(x)
        self.assertTrue('Not appropriate input shape.' in str(context.exception))

        x = np.ones((2,10,1))
        with self.assertRaises(Exception) as context:
            output = utils.atmost_1d(x)
        self.assertTrue('Not appropriate input shape.' in str(context.exception))

    def test_make_grid(self):
        """
        Test that the make grid function returns a symmetric, uniform grid of
        points. Make grid is primarily used in the GPViz module.
        """
        true = (np.array([[-1, -1],
                          [-1,  0],
                          [ 0, -1],
                          [ 0,  0]]), 2)

        output = utils.make_grid((-1,1,1))
        npt.assert_almost_equal(output[0], true[0], decimal=10)
        assert output[1] == true[1]

        with self.assertRaises(Exception) as context:
            output = utils.make_grid((2,1,1))
        self.assertTrue('Min value greater than max value.' in str(context.exception))

        with self.assertRaises(Exception) as context:
            output = utils.make_grid((-2,1,1))
        self.assertTrue('Plot topology not square!' in str(context.exception))

    def test_invert(self):
        """
        Test that inversion methods work on a typical matrix input with a
        reasonable condition.
        """
        arr = np.random.rand(5, 5)
        arr = arr.dot(arr.T)
        true = np.array([[  52.02885046,  -53.32959006,   84.43016798,   -0.88702483,  -67.39164907],
                         [ -53.32959006,  109.01905249, -126.15797593,  -22.1530339,    67.13403575],
                         [  84.43016798, -126.15797593,  179.4915856,    12.99237804, -119.02932256],
                         [  -0.88702483,  -22.1530339,    12.99237804,   10.8927777,     3.57616868],
                         [ -67.39164907,   67.13403575, -119.02932256,    3.57616868,   96.96479211]])

        output = utils.invert(arr,"inv")
        npt.assert_almost_equal(output, true, decimal=7)

        output = utils.invert(arr,"pinv")
        npt.assert_almost_equal(output, true, decimal=7)

        output = utils.invert(arr,"solve")
        npt.assert_almost_equal(output, true, decimal=7)

        output = utils.invert(arr,"cholesky")
        npt.assert_almost_equal(output, true, decimal=7)

        output = utils.invert(arr,"svd")
        npt.assert_almost_equal(output, true, decimal=7)

        output = utils.invert(arr,"lu")
        npt.assert_almost_equal(output, true, decimal=7)

        with self.assertRaises(Exception) as context:
            output = utils.invert(arr,"fake")
        self.assertTrue('Invalid inversion method argument.' in str(context.exception))

        arr = np.random.rand(10, 10)
        arr[-1] = arr[0] + arr[1]

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            utils.invert(arr)
            # Verify some things
            assert len(w) == 1
            assert "Matrix has high condition." in str(w[-1].message)

    def test_onehot(self):
        """
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

    def test_reversehot(self):
        """
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

    def test_deprecated(self):
        """
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
