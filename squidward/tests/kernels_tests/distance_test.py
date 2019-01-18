import unittest
import numpy as np
from squidward.kernels import distance
import numpy.testing as npt
np.random.seed(0)

# useful for debugging
np.set_printoptions(suppress=True)

class distanceTestCase(unittest.TestCase):
    """Class for distance function tests."""

    def setUp(self):
        """
        Set Up
        Set up shared environment or variables for tests
        """
        pass

class RBFTestCase(distanceTestCase):
    """Tests for radial basis function."""

    def test_normal_input(self):
        """
        Normal Input
        Test case for expected inputs
        """
        d = distance.RBF(12.0,14.0**2)

        a = np.random.rand(10,11,12)
        b = np.random.rand(10,11,12)

        true = 89.7086004545091
        output = d(a, b)
        self.assertEqual(output, true)

    def test_params_assertions(self):
        """
        Params Assertions
        Test case for invalid kernal parameter arguments.
        """
        with self.assertRaises(Exception) as context:
            distance.RBF(0.0,14.0**2)
        self.assertTrue('Lengthscale parameter must be greater than zero' in str(context.exception))

        with self.assertRaises(Exception) as context:
            distance.RBF(12.0,0.0**2)
        print( str(context.exception) )
        self.assertTrue('Kernel variance parameter must be greater than zero' in str(context.exception))

if __name__ == '__main__':
    unittest.main()
