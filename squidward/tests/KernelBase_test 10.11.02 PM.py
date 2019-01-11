import unittest
import numpy as np
from squidward.Kernels import KernelBase
import numpy.testing as npt
np.random.seed(0)

# useful for debugging
np.set_printoptions(suppress=True)

class KernelBaseTestCase(unittest.TestCase):
    """
    Class for kernel base tests.
    """
    def setUp(self):
        """
        Set up shared environment or variables for tests.
        """
        def test_distance(alpha, beta):
            return np.sum(alpha+beta)

        alpha = np.array([[[0.80041543, 0.38054451, 0.40149714, 0.72283192],
                           [0.2079518 , 0.45512953, 0.36712486, 0.00935587],
                           [0.89406573, 0.18693555, 0.85264612, 0.05814072]],
                          [[0.62308415, 0.05840852, 0.20080878, 0.0621183 ],
                           [0.18278032, 0.52933488, 0.39774963, 0.21107714],
                           [0.15100123, 0.10334189, 0.70711461, 0.91012337]],
                          [[0.97002337, 0.26950622, 0.71561766, 0.99829289],
                           [0.48554913, 0.41227646, 0.52769877, 0.84069773],
                           [0.76406377, 0.02587738, 0.05878135, 0.22412919]],
                          [[0.97068207, 0.35700262, 0.08605499, 0.79074753],
                           [0.42397823, 0.51037705, 0.44919402, 0.41989193],
                           [0.19562684, 0.73708793, 0.74242757, 0.59625942]],
                          [[0.23821979, 0.38966354, 0.90705623, 0.40568901],
                           [0.53806061, 0.92896669, 0.59246232, 0.44687328],
                           [0.56047209, 0.14184412, 0.02513565, 0.02057055]]])

        beta = np.array([[[0.85374384, 0.9351648 , 0.99778116, 0.59917667],
                          [0.76249341, 0.11790279, 0.63020352, 0.74818649],
                          [0.30539085, 0.09604326, 0.83380014, 0.95491305]],
                         [[0.2600214 , 0.35249871, 0.97274905, 0.02612635],
                          [0.00537489, 0.71621324, 0.18989205, 0.05632158],
                          [0.69679991, 0.15653632, 0.03538905, 0.97532232]],
                         [[0.27835309, 0.11407254, 0.18947242, 0.30772099],
                          [0.49571862, 0.43644311, 0.72362702, 0.65960954],
                          [0.19770792, 0.70667906, 0.37997083, 0.30715239]],
                         [[0.85508522, 0.91198967, 0.885982  , 0.22832638],
                          [0.39889572, 0.01076971, 0.68855572, 0.74966257],
                          [0.57711358, 0.62980682, 0.33804398, 0.10674211]],
                         [[0.49607274, 0.46164375, 0.41259604, 0.36971801],
                          [0.9796552 , 0.43068297, 0.49410799, 0.62269753],
                          [0.53705091, 0.94835502, 0.2834392 , 0.5570391 ]]])

        self.dist = test_distance
        self.alpha = alpha
        self.beta = beta

    def test_params_assertions(self):
        """
        Test that the KernelBase assertions work to raise exceptions for invalid parameters.
        """
        d = self.dist

        with self.assertRaises(Exception) as context:
            KernelBase.Kernel(d, 'fake')
        self.assertTrue('Invalid argument for kernel method' in str(context.exception))

class KOneTestCase(KernelBaseTestCase):
    """
    Test that the k1 method of KernelBase returns valid kernels.
    """
    def test_normal_input(self):
        """
        Test that k1 normal inputs return expected result.
        """
        d = self.dist
        a = self.alpha
        b = self.beta

        kernel = KernelBase.Kernel(d, 'k1')
        output = kernel.k(a, b)
        true = np.array([[13.17143916,  9.77988405, 10.13316671, 11.71761266, 11.92969764],
                         [11.9717428,   8.58018769,  8.93347035, 10.5179163,  10.73000128],
                         [14.1273139,  10.73575879, 11.08904145, 12.6734874,  12.88557238],
                         [14.11413018, 10.72257507, 11.07585773, 12.66030368, 12.87238866],
                         [13.02981386,  9.63825875,  9.99154141, 11.57598736, 11.78807234]])
        npt.assert_almost_equal(output, true, decimal=7)

if __name__ == '__main__':
    unittest.main()
