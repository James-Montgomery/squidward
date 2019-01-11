import unittest
import numpy as np
from squidward.Kernels import Distance
import numpy.testing as npt
np.random.seed(0)

# useful for debugging
np.set_printoptions(suppress=True)

class DistanceTestCase(unittest.TestCase):
    """
    Class for distance function tests.
    """
    def setUp(self):
        """
        Set up shared environment or variables for tests
        """
        pass

class RBFTestCase(DistanceTestCase):
    """
    Tests for radial basis function.
    """
    def test_normal_input(self):
        """
        Test case for expected inputs
        """
        d = Distance.rbf(12.0,14.0**2)

        a = np.random.rand(10,11,12)
        b = np.random.rand(10,11,12)

        true = 89.7086004545091
        output = d(a, b)
        self.assertEqual(output, true)

    def test_params_assertions(self):
        """
        Test case for invalid kernal parameter arguments.
        """
        with self.assertRaises(Exception) as context:
            Distance.rbf(0.0,14.0**2)
        self.assertTrue('Lengthscale parameter must be greater than zero' in str(context.exception))

        with self.assertRaises(Exception) as context:
            Distance.rbf(12.0,0.0**2)
        self.assertTrue('Kernel variance parameter must be greater than zero' in str(context.exception))

if __name__ == '__main__':
    unittest.main()
