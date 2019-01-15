from squidward import validation
import unittest
import numpy as np
import numpy.testing as npt
np.random.seed(0)

class validationTestCase(unittest.TestCase):
    """
    Class for validation tests.
    """
    def test_preprocess(self):
        """
        Test classification validation preprocessing code.
        """
        preprocess = validation.preprocess

        @preprocess
        def f(prediction, target):
            return True

        prediction = np.ones(10)
        target = np.zeros(9)

        with self.assertRaises(Exception) as context:
           f(prediction, target)
        self.assertTrue('Number of predictions does not match number of targets' in str(context.exception))

    def test_likelihood(self):
        """
        Test that the likelihood function returns the correct likeliohood or
        log likelihood.
        """
        means = np.linspace(0, 4, 5).reshape(-1, 1)
        cov = np.random.rand(5, 5)
        cov = cov.dot(cov.T)
        prediction = np.array([1.2815938,  2.46052947, 3.49448986, 3.71525343, 5.5182648 ])

        true = -1.941715114406031
        output = validation.likelihood(means, cov, prediction, True)
        self.assertEqual(output, true)

        true = 0.14345769230347272
        output = validation.likelihood(means, cov, prediction, False)
        self.assertEqual(output, true)

    def test_rmse(self):
        """
        Test that rmse returns the correct root mean squared error.
        """
        prediction = np.random.rand(10)
        target = np.random.rand(10) + 1.0
        output = validation.rmse(prediction=prediction, target=target)
        true = 1.1545450663694088
        self.assertEqual(output, true)

    def test_acc(self):
        """
        Test that acc returns the correct accuracy.
        """
        target = np.array([0,1,2,1,0,1,0,2,1]).reshape(-1,1)
        prediction = np.array([1,2,2,1,1,1,2,2,1]).reshape(-1,1)
        output = validation.acc(prediction, target)
        true = 0.5555555555555556
        self.assertEqual(output, true)

        target = np.array([0,1,2,1,0,1,0,2,1])
        predictions = np.array([1,2,2,1,1,1,2,2,1])
        output = validation.acc(prediction, target)
        true = 0.5555555555555556
        self.assertEqual(output, true)

if __name__ == '__main__':
    unittest.main()
