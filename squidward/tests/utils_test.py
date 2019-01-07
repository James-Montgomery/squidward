import unittest
import warnings
import numpy as np
import numpy.testing as npt
from squidward import utils
np.random.seed(0)

class TestUtilities(unittest.TestCase):

    def test_sigmoid(self):
        """
        Test sigmoid functions works
        """
        x = np.array([[-8,0,6],[8,3,1],[10,-300,11]])
        output = utils.sigmoid(x)
        true = np.array([[3.3535013047e-004, 5.0000000000e-001, 9.9752737684e-001],
                         [9.9966464987e-001, 9.5257412682e-001, 7.3105857863e-001],
                         [9.9995460213e-001, 5.1482002224e-131, 9.9998329858e-001]])
        npt.assert_almost_equal(output, true, decimal=10)

    def test_softmax(self):
        """
        Test softmax function works
        """
        x = np.array([[-8,0,6],[8,3,1],[10,-300,11]])
        x = utils.sigmoid(x)
        output = utils.softmax(x)
        true = np.array([[2.2388575697e-004, 3.3380896059e-001, 6.6596715365e-001],
                         [3.7255082739e-001, 3.5500132884e-001, 2.7244784376e-001],
                         [4.9999282567e-001, 2.5741800386e-131, 5.0000717433e-001]])
        npt.assert_almost_equal(output, true, decimal=10)

    def test_is_invertible_true(self):
        """
        Test that non-singular matricies return true.
        """
        arr = np.random.rand(10, 10)
        arr = arr.dot(arr.T)
        output = utils.is_invertible(arr)
        assert output

    def test_is_invertible_false(self):
        """
        Test that singular matricies return false.
        """
        arr = np.random.rand(10, 10)
        arr[-1] = arr[0] + arr[1]
        output = utils.is_invertible(arr)
        assert ~output

    def test_check_valid_cov(self):
        """
        Test that the function that validates covariance matricies works
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
        """
        x = np.ones(10)
        output = utils.atleast_2d(x)
        true = x = np.ones(10).reshape(-1,1)
        npt.assert_almost_equal(output, true, decimal=10)

        x = np.ones((10,10,2))
        output = utils.atleast_2d(x)
        npt.assert_almost_equal(output, x, decimal=10)

    def test_atmost_1d(self):
        """
        """
        x = np.ones((10))
        output = utils.atmost_1d(x)
        true = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
        npt.assert_almost_equal(output, true, decimal=10)

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

    def test_makegrid(self):
        """
        """
        output = utils.make_grid((-1,1,1))
        true = (np.array([[-1, -1],
                          [-1,  0],
                          [ 0, -1],
                          [ 0,  0]]), 2)
        npt.assert_almost_equal(output[0], true[0], decimal=10)

        assert output[1] == true[1]

        with self.assertRaises(Exception) as context:
            output = utils.make_grid((2,1,1))
        self.assertTrue('Min value greater than max value.' in str(context.exception))

        with self.assertRaises(Exception) as context:
            output = utils.make_grid((-2,1,1))
        self.assertTrue('Plot topology not square!' in str(context.exception))

    def test_invert(self):
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

    def test_onehot(self):
        y = np.random.randint(0,3,10)
        output = utils.onehot(y,3)
        true = np.array([[1., 0., 0.],
                         [1., 0., 0.],
                         [1., 0., 0.],
                         [1., 0., 0.],
                         [0., 1., 0.],
                         [1., 0., 0.],
                         [0., 0., 1.],
                         [0., 0., 1.],
                         [0., 0., 1.],
                         [1., 0., 0.]])

        npt.assert_almost_equal(output, true, decimal=10)

        with self.assertRaises(Exception) as context:
            utils.onehot(y,4)
        self.assertTrue('Number of unique values does not match num_classes argument.' in str(context.exception))

        with self.assertRaises(Exception) as context:
            utils.onehot(y,2)
        self.assertTrue('Number of unique values does not match num_classes argument.' in str(context.exception))

    def test_reversehot(self):
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
        output = utils.reversehot(y)
        true = np.array([0.0,0.0,0.0,0.0,1.0,0.0,2.0,2.0,2.0,0.0])

        npt.assert_almost_equal(output, true, decimal=10)

        output = utils.reversehot(true)

        npt.assert_almost_equal(output, true, decimal=10)

        output = utils.reversehot(true.T)

        npt.assert_almost_equal(output, true, decimal=10)

if __name__ == '__main__':
    unittest.main()
