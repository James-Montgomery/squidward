import unittest

import numpy as np

from squidward import validation


class ValidationTestCase(unittest.TestCase):
    """Class for validation tests."""

    def test_preprocess(self):
        """
        Preprocess
        Test classification validation preprocessing code.
        """
        preprocess = validation.preprocess

        @preprocess
        def test_function(predictions, targets):
            return True

        predictions = np.ones(10)
        targets = np.zeros(9)

        with self.assertRaises(Exception) as context:
           test_function(predictions, targets)
        self.assertTrue('Number of predictions does not match number of targets' in str(context.exception))

    def test_rmse(self):
        """
        Root Mean Squared Error
        Test that rmse returns the correct root mean squared error.
        """
        predictions = np.array([0.76392998, 0.75568339, 0.41298595, 0.74049558, 0.92748847,
                                0.72007371, 0.52249059, 0.59100948, 0.86575088, 0.19507582])

        targets = np.array([0.18075874, 0.58670919, 0.60749056, 0.81186994, 0.20804091,
                            0.1987932 , 0.92317227, 0.26883039, 0.24775426, 0.10320547]) + 1.0

        output = validation.rmse(predictions=predictions, targets=targets)
        true = 0.844919688370596
        self.assertEqual(output, true)

        targets = targets.reshape(-1, 1)
        predictions = predictions.reshape(-1, 1)

        output = validation.rmse(predictions=predictions, targets=targets)
        self.assertEqual(output, true)

    def test_acc(self):
        """
        Accuracy
        Test that acc returns the correct accuracy.
        """

        targets = np.array([0, 1, 2, 1, 0, 1, 0, 2, 1])
        predictions = np.array([1, 2, 2, 1, 1, 1, 2, 2, 1])

        output = validation.acc(predictions=predictions, targets=targets)
        true = 0.5555555555555556
        self.assertEqual(output, true)

        targets = targets.reshape(-1,1)
        predictions = predictions.reshape(-1,1)

        output = validation.acc(predictions=predictions, targets=targets)
        self.assertEqual(output, true)


if __name__ == '__main__':
    unittest.main()
