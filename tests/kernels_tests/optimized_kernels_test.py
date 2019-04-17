import unittest
import numpy as np
import numpy.testing as npt

from squidward.kernels import optimized_kernels

# useful for debugging
np.set_printoptions(suppress=True)


class OptimizedKernelsTestCase(unittest.TestCase):
    """Class for optimized kernel tests."""

    def setUp(self):
        """
        Set Up
        Set up shared environment or variables for tests
        """

        seed = 0
        random = np.random.RandomState(seed)

        self.alpha = random.normal(0, 1, (5, 3))
        self.beta = random.normal(0, 1, (5, 3))


class RBFTestCase(OptimizedKernelsTestCase):
    """Tests for radial basis function."""

    def test_1(self):
        """
        Parameter Checks
        Ensure exceptions are thrown for invalid parameters.
        """

        # pass an invalid lengthscale
        with self.assertRaises(Exception) as context:
            optimized_kernels.RBF_Kernel(lengthscale=0.0, var_k=1.0)
        self.assertTrue("Lengthscale parameter must be greater than zero." in str(context.exception))

        # pass an invalid kernel variance
        with self.assertRaises(Exception) as context:
            optimized_kernels.RBF_Kernel(lengthscale=1.0, var_k=0.0)
        self.assertTrue("Kernel variance parameter must be greater than zero." in str(context.exception))

    def test_2(self):
        """
        Integration Test
        """

        kernel = optimized_kernels.RBF_Kernel(lengthscale=0.5, var_k=2.0)

        output = kernel(self.alpha, self.alpha)

        expected_output = np.array([[2.        , 0.00000813, 0.02783322, 0.02861043, 0.12922294],
                                    [0.00000813, 2.        , 0.00000447, 0.        , 0.00000099],
                                    [0.02783322, 0.00000447, 2.        , 0.00733543, 0.88159373],
                                    [0.02861043, 0.        , 0.00733543, 2.        , 0.20284226],
                                    [0.12922294, 0.00000099, 0.88159373, 0.20284226, 2.        ]])

        npt.assert_almost_equal(output, expected_output, decimal=7)

        output = kernel(self.alpha, self.beta)

        expected_output = np.array([[0.00018496, 0.         ,0.00029542, 0.00021654, 0.00004685],
                                    [0.0003181 , 0.         ,0.00155109, 0.        , 0.        ],
                                    [0.00407661, 0.00000203 ,0.09415448, 0.00196957, 0.00000368],
                                    [0.00020939, 0.         ,0.00004061, 0.00000023, 0.020668  ],
                                    [0.01382087, 0.         ,0.03890452, 0.00010685, 0.00075348]])

        npt.assert_almost_equal(output, expected_output, decimal=7)


if __name__ == '__main__':
    unittest.main()