from squidward import Validation
import unittest
import numpy as np
import numpy.testing as npt
np.random.seed(0)

class ValidationTestCase(unittest.TestCase):
    """
    Class for validation tests.
    """
    def test_likelihood(self):
        """
        Test that the likelihood function returns the correct likeliohood or
        log likelihood.
        """
        means = np.linspace(0,4,5).reshape(-1,1)
        cov = np.random.rand(5,5)
        cov = cov.dot(cov.T)
        y = np.array([1.2815938,  2.46052947, 3.49448986, 3.71525343, 5.5182648 ])

        true = -1.941715114406031
        output = Validation.likelihood(means, cov, y, True)
        self.assertEqual(output, true)

        true = 0.14345769230347272
        output = Validation.likelihood(means, cov, y, False)
        self.assertEqual(output, true)

    def test_rmse(self):
        """
        Test that rmse returns the correct root mean squared error.
        """
        x = np.random.rand(10)
        y = np.random.rand(10) + 1.0
        output = Validation.rmse(p=x,y=y)
        true = 1.1545450663694088
        self.assertEqual(output, true)

    def test_acc(self):
        """
        Test that acc returns the correct accuracy.
        """
        y = np.array([0,1,2,1,0,1,0,2,1]).reshape(-1,1)
        p = np.array([1,2,2,1,1,1,2,2,1]).reshape(-1,1)
        output = Validation.acc(y,p)
        true = 0.5555555555555556
        self.assertEqual(output, true)

        y = np.array([0,1,2,1,0,1,0,2,1])
        p = np.array([1,2,2,1,1,1,2,2,1])
        output = Validation.acc(y,p)
        true = 0.5555555555555556
        self.assertEqual(output, true)

    def test_brier(self):
        """
        """
        try:
            Validation.brier_score()
        except:
            pass

    def test_precision(self):
        """
        """
        try:
            Validation.precision()
        except:
            pass

    def test_recall(self):
        """
        """
        try:
            Validation.recall()
        except:
            pass

    def test_roc_auc(self):
        """
        """
        try:
            Validation.roc_auc()
        except:
            pass

    def test_posterior_checks(self):
        """
        """
        try:
            Validation.posterior_checks()
        except:
            pass

if __name__ == '__main__':
    unittest.main()
