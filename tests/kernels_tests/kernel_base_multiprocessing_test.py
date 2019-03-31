import unittest
import numpy as np
from squidward.kernels import kernel_base_multiprocessing as kernel_base
import numpy.testing as npt
np.random.seed(0)

# useful for debugging
np.set_printoptions(suppress=True)

class kernel_baseTestCase(unittest.TestCase):
    """Class for kernel base tests."""

    def setUp(self):
        """
        Set Up
        Set up shared environment or variables for tests.
        """
        def test_distance(alpha, beta):
            return np.sum(np.abs(alpha+beta))

        alpha = np.array([[0.74775167, 0.09963786, 0.24078391,
                           0.16182137, 0.72764008, 0.13583729],
                          [0.1427068 , 0.61315893, 0.61929491,
                           0.88585838, 0.96463067, 0.81522606],
                          [0.16291094, 0.59166813, 0.17442026,
                           0.34366403, 0.75932132, 0.02574967],
                          [0.16449091, 0.71445737, 0.71023514,
                           0.22225375, 0.49399307, 0.62012281],
                          [0.36558453, 0.50613319, 0.62597648,
                          0.44873197, 0.5863159 , 0.44340289]])

        beta = np.array([[0.83951641, 0.1409072 , 0.96615312,
                          0.1480484 , 0.17059111, 0.06294672],
                         [0.10287028, 0.1472429 , 0.99437229,
                          0.25818537, 0.907556  , 0.11534158],
                         [0.03248681, 0.09601429, 0.88024617,
                          0.42245145, 0.0778328 , 0.22152001],
                         [0.45315421, 0.13692575, 0.44841676,
                          0.2994347 , 0.56941712, 0.80653022],
                         [0.81958843, 0.79307846, 0.80562772,
                          0.03878067, 0.74121989, 0.1833349 ]])

        self.dist = test_distance
        self.alpha = alpha
        self.beta = beta

    def test_unequal_features(self):
        """
        Unequal Features
        Test that exception is raised if arrays with unequal numbers of
        features are passed.
        """
        d = self.dist
        a = np.random.rand(10,2)
        b = np.random.rand(10,3)

        kernel = kernel_base.Kernel(d)

        with self.assertRaises(Exception) as context:
            kernel(a,b)
        self.assertTrue('Input arrays have differing number of features.' in str(context.exception))

class KernelTestCase(kernel_baseTestCase):
    """Test that the k1 method of kernel_base returns valid kernels."""

    # TODO: fix test. Multiprocessing not playing nice
    # with unit tests.
    def test_normal_input(self):
        """
        Normal Input
        Test that normal inputs return expected result.
        """
        # d = self.dist
        # a = self.alpha
        # b = self.beta
        #
        #kernel = kernel_base.Kernel(d)
        #output = kernel(a, b)
        # true = np.array([[4.44163514, 4.6390406,  3.84402371, 4.82735094, 5.49510225],
        #                  [6.36903871, 6.56644417, 5.77142728, 6.75475451, 7.42250582],
        #                  [4.38589731, 4.58330277, 3.78828588, 4.77161311, 5.43936442],
        #                  [5.25371601, 5.45112147, 4.65610458, 5.63943181, 6.30718312],
        #                  [5.30430792, 5.50171338, 4.70669649, 5.69002372, 6.35777503]])
        #npt.assert_almost_equal(output, true, decimal=7)
