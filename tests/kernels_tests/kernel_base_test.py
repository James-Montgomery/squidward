import unittest
import numpy as np
import numpy.testing as npt

from squidward.kernels import kernel_base

# useful for debugging
np.set_printoptions(suppress=True)


# ---------------------------------------------------------------------------------------------------------------------
# kernel class tests
# ---------------------------------------------------------------------------------------------------------------------

class KernelTestCase(unittest.TestCase):
    """Class for kernel base tests."""

    def setUp(self):
        """
        Set Up
        Set up shared environment or variables for tests.
        """

        seed = 0
        random = np.random.RandomState(seed)

        # a simple differencing function for testing
        def test_distance(input_a, input_b):
            return np.exp( 1.0/(2.0*1.0**2.0) * np.sum( np.abs(input_a - input_b) ) )

        self.dist = test_distance
        self.alpha = random.normal(0, 1, (5, 3))
        self.beta = random.normal(0, 1, (5, 3))

    def test_1(self):
        """
        Params Assertions
        Test that the kernel_base assertions work to raise exceptions for invalid parameters.
        """
        d = self.dist

        with self.assertRaises(Exception) as context:
            kernel_base.Kernel(method='k1')
        self.assertTrue("Model object must be instantiated with a valid distance function." in str(context.exception))

        with self.assertRaises(Exception) as context:
            kernel_base.Kernel(distance_function='grabage', method='k1')
        self.assertTrue("Model object must be instantiated with a valid distance function." in str(context.exception))

        with self.assertRaises(Exception) as context:
            kernel_base.Kernel(d, 'fake')
        self.assertTrue('Invalid argument for kernel method' in str(context.exception))

    def test_2(self):
        """
        K1
        Test that the first kernel function option behaves as expected.
        """
        d = self.dist
        a = self.alpha
        b = self.beta

        kernel = kernel_base.Kernel(d, 'k1')
        output = kernel(a, a)

        expected_output = np.array([[ 1.        ,  7.02959258,  3.39980335,  2.83644498,  2.47981952],
                                    [ 7.02959258,  1.        ,  8.10020609, 19.93905261, 10.21073666],
                                    [ 3.39980335,  8.10020609,  1.        ,  3.30749639,  1.65629387],
                                    [ 2.83644498, 19.93905261,  3.30749639,  1.        ,  1.99692606],
                                    [ 2.47981952, 10.21073666,  1.65629387,  1.99692606,  1.        ]])

        npt.assert_almost_equal(output, expected_output, decimal=7)

        output = kernel(a, b)

        expected_output = np.array([[ 6.38593596, 22.61204426,  5.19557797,  5.18931817,  5.97294135],
                                    [ 4.60162132, 22.47931564,  4.10725379,  8.90765254, 13.52779202],
                                    [ 3.26054674,  6.65098594,  2.65277085,  3.99812708,  8.99791745],
                                    [ 4.67952086, 12.82590196,  4.85459473, 11.39346766,  2.72046176],
                                    [ 3.4021054 ,  9.11842338,  2.76794255,  5.70550303,  5.43256096]])

        npt.assert_almost_equal(output, expected_output, decimal=7)

    def test_3(self):
        """
        K1
        Test that the first kernel function option behaves as expected.
        """
        d = self.dist
        a = self.alpha
        b = self.beta

        kernel = kernel_base.Kernel(d, 'k2')
        output = kernel(a, a)

        expected_output = np.array([[ 1.        ,  7.02959258,  3.39980335,  2.83644498,  2.47981952],
                                    [ 7.02959258,  1.        ,  8.10020609, 19.93905261, 10.21073666],
                                    [ 3.39980335,  8.10020609,  1.        ,  3.30749639,  1.65629387],
                                    [ 2.83644498, 19.93905261,  3.30749639,  1.        ,  1.99692606],
                                    [ 2.47981952, 10.21073666,  1.65629387,  1.99692606,  1.        ]])

        npt.assert_almost_equal(output, expected_output, decimal=7)

        output = kernel(a, b)

        expected_output = np.array([[ 6.38593596, 22.61204426,  5.19557797,  5.18931817,  5.97294135],
                                    [ 4.60162132, 22.47931564,  4.10725379,  8.90765254, 13.52779202],
                                    [ 3.26054674,  6.65098594,  2.65277085,  3.99812708,  8.99791745],
                                    [ 4.67952086, 12.82590196,  4.85459473, 11.39346766,  2.72046176],
                                    [ 3.4021054 ,  9.11842338,  2.76794255,  5.70550303,  5.43256096]])

        npt.assert_almost_equal(output, expected_output, decimal=7)


if __name__ == '__main__':
    unittest.main()
