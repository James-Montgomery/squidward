from squidward import validation
import unittest
import numpy as np
import numpy.testing as npt
np.random.seed(0)

class TestUtilities(unittest.TestCase):

    def test_rmse(self):
        """
        """
        x = np.random.rand(10)
        y = np.random.rand(10) + 1.0
        output = validation.rmse(x,y)
        true = 0.9877661052357045
        assert output

    def test_likelihood(self):
        means = np.linspace(0,4,5).reshape(-1,1)
        cov = np.random.rand(5,5)
        cov = cov.dot(cov.T)
        y = means + 1.0

        output = validation.likelihood(means, cov, y)
        true = np.array([2.78725618e-25, 9.61388482e-23, 5.51482346e-23,
                         5.26108097e-26, 8.34697340e-32])
        npt.assert_almost_equal(output, true, decimal=10)

        output = validation.likelihood(means, cov, y, True)
        true = np.array([-877.3227103867, -665.7262049664, -486.1115914062,
                         -338.4788697063, -222.8280398666])
        npt.assert_almost_equal(output, true, decimal=10)

    def test_acc(self):
        y = np.random.randint(0,3,10).reshape(-1,1)
        p = np.random.randint(0,3,10).reshape(-1,1)
        validation.acc(y,p)


if __name__ == '__main__':
    unittest.main()
