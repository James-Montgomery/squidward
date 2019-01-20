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

        a = np.array([0.70908325, 0.97063918, 0.2901618 , 0.20842557, 0.12981446,
                      0.86155565, 0.71940193, 0.34456898, 0.65642592, 0.70362763])
        b = np.array([0.89592768, 0.83254226, 0.67318587, 0.06753976, 0.4851254 ,
                      0.60731089, 0.29373965, 0.59003959, 0.40602443, 0.34742655])

        true = 195.42750477495602
        output = d(a, b)
        self.assertEqual(output, true)

        a = a.reshape(-1, 1)
        b = b.reshape(-1, 1)

        true = 195.42750477495602
        output = d(a, b)
        self.assertEqual(output, true)

        a = np.random.rand(10,2)
        with self.assertRaises(Exception) as context:
            output = d(a, a)
        self.assertTrue('Not appropriate input shape.' in str(context.exception))

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
