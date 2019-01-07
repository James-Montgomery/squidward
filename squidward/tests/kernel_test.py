import unittest
import numpy as np
import numpy.testing as npt
from squidward import kernels
np.random.seed(0)

class TestUtilities(unittest.TestCase):

    def test_RBF(self):
        alpha = np.random.rand(3)
        beta = np.random.rand(3)
        kernel = kernels.RBF(10,0.1)
        output = kernel.k(alpha, beta)
        true = np.array([[9.99227927, 4.56925433, 6.24232594],
                         [2.34521079, 0.14269653, 7.86555804],
                         [8.4577168,  2.01091729, 9.11181633]])

        npt.assert_almost_equal(output, true, decimal=7)

        alpha = np.random.rand(3,3)
        beta = np.random.rand(3,3)
        kernel = kernels.RBF(10,0.1)
        output = kernel.k(alpha, beta)
        true = np.array([[2.92028397e-17, 2.92564364e+00, 2.58431977e-05],
                         [1.46277258e-14, 1.23574448e-05, 3.19167986e-05],
                         [3.83093044e-33, 1.22384588e-16, 1.70573471e-14]])
        
        npt.assert_almost_equal(output, true, decimal=7)
