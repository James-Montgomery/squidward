import unittest
from unittest import mock
import numpy as np
import numpy.testing as npt

from squidward import gpr
from squidward.kernels import distance, kernel_base

# useful for debugging
np.set_printoptions(suppress=True)

# TODO: ensure only using double quotes
# TODO: ensure all outputs are either single or double dimensional

# ---------------------------------------------------------------------------------------------------------------------
# Set up data and model parameters for test
# ---------------------------------------------------------------------------------------------------------------------

class RegressionTestCase(unittest.TestCase):
    """Class for guassian process regression tests."""

    def setUp(self):
        """
        Set Up
        Set up shared environment or variables for tests.
        """

        # train data
        x_train = np.array([[0.74775167, 0.09963786, 0.24078391, 0.16182137, 0.72764008, 0.13583729],
                            [0.1427068 , 0.61315893, 0.61929491, 0.88585838, 0.96463067, 0.81522606],
                            [0.16291094, 0.59166813, 0.17442026, 0.34366403, 0.75932132, 0.02574967],
                            [0.16449091, 0.71445737, 0.71023514, 0.22225375, 0.49399307, 0.62012281],
                            [0.36558453, 0.50613319, 0.62597648, 0.44873197, 0.5863159 , 0.44340289]])

        y_train = np.array([0.63095398, 0.29118303, 0.85576359, 0.63095398, 0.29118303])

        # create distance function
        d = distance.RBF(3.0, 2.0**2)

        # create kernel
        kernel = kernel_base.Kernel(d, 'k1')

        self.x_train = x_train
        self.y_train = y_train
        self.kernel = kernel


# ---------------------------------------------------------------------------------------------------------------------
# Tests for base class
# ---------------------------------------------------------------------------------------------------------------------


class GaussianProcessBaseClassTestCase(RegressionTestCase):
    """Tests for Gaussian Process Base Class"""

    def test_1(self):
        """
        Parameter Checks
        This test verifies that all parameter checks through the proper exceptions.
        """

        # should not throw an exception
        gpr.GaussianProcessBase(kernel=self.kernel)
        gpr.GaussianProcessBase(kernel=self.kernel, var_l=np.array([1, 2, 3, 4, 5]))
        gpr.GaussianProcessBase(kernel=self.kernel, var_l=[1, 2, 3, 4, 5])
        gpr.GaussianProcessBase(kernel=self.kernel, var_l=1.0)
        gpr.GaussianProcessBase(kernel=self.kernel, var_l=1)
        gpr.GaussianProcessBase(kernel=self.kernel, seed=10)
        gpr.GaussianProcessBase(kernel=self.kernel, show_warnings=False)

        likelihood_exception = "Likelihood variance argument must be a positive integer, " \
                               "float, array of integers, or array of floats."

        # specify a non numeric likelihood variance
        with self.assertRaises(Exception) as context:
            gpr.GaussianProcessBase(kernel=self.kernel, var_l="a")
        self.assertTrue(
            likelihood_exception in str(context.exception))

        # specify a negative likelihood variance
        with self.assertRaises(Exception) as context:
           gpr.GaussianProcessBase(kernel=self.kernel, var_l=-1.0)
        self.assertTrue(
            likelihood_exception in str(context.exception))

        # specify a likelihood variance with too many dimensions
        with self.assertRaises(Exception) as context:
            gpr.GaussianProcessBase(kernel=self.kernel, var_l=np.ones((3, 3)))
        self.assertTrue("Error with likelihood variance argument." in str(context.exception))

        # specify an empty array for likelihood variance
        with self.assertRaises(Exception) as context:
            gpr.GaussianProcessBase(kernel=self.kernel, var_l=[])
        self.assertTrue(
            likelihood_exception in str(context.exception))

        # specify a non numeric random seed
        with self.assertRaises(Exception) as context:
            gpr.GaussianProcessBase(kernel=self.kernel, seed="a")
        self.assertTrue("When specifying random seed argument it must be an integer." in str(context.exception))

        # specify a non boolean for show warnings
        with self.assertRaises(Exception) as context:
            gpr.GaussianProcessBase(kernel=self.kernel, show_warnings="a")
        self.assertTrue("Show warnings argument must be a boolean." in str(context.exception))

        # don't specify kernel
        with self.assertRaises(Exception) as context:
            gpr.GaussianProcessBase(var_l=1050 ** 2)
        self.assertTrue("Model object must be instantiated with a valid kernel object." in str(context.exception))

    def test_2(self):
        """
        Prior Predict
        This test verifies that the prior predict function returns valid predictions.
        """

        with mock.patch('squidward.gpr.check_valid_cov') as mock_check_valid_cov:

            model = gpr.GaussianProcessBase(kernel=self.kernel, var_l=1.0, seed=0)
            mean, var = model.prior_predict(self.x_train, False)

            mock_check_valid_cov.assert_called_once()

        expected_mean = np.array([0., 0., 0., 0., 0.]).reshape(-1, 1)
        expected_var = np.array([4., 4., 4., 4., 4.]).reshape(-1, 1)

        npt.assert_equal(mean, expected_mean)
        npt.assert_equal(var, expected_var)

        with mock.patch('squidward.gpr.check_valid_cov') as mock_check_valid_cov:

            model = gpr.GaussianProcessBase(kernel=self.kernel, var_l=1.0, seed=0)
            mean, cov = model.prior_predict(self.x_train, True)

            mock_check_valid_cov.assert_called_once()

        expected_mean = np.array([0., 0., 0., 0., 0.]).reshape(-1, 1)
        expected_cov = np.array([[4.        , 3.61633139, 3.86140763, 3.73551499, 3.85672873],
                                 [3.61633139, 4.        , 3.75073137, 3.84338362, 3.8831566 ],
                                 [3.86140763, 3.75073137, 4.        , 3.83876279, 3.89740527],
                                 [3.73551499, 3.84338362, 3.83876279, 4.        , 3.95976309],
                                 [3.85672873, 3.8831566 , 3.89740527, 3.95976309, 4.        ]])

        npt.assert_almost_equal(mean, expected_mean, decimal=7)
        npt.assert_almost_equal(cov, expected_cov, decimal=7)

    def test_3(self):
        """
        Prior Sample
        This test verifies that the prior sample function returns valid samples. Consistent
        sampling behavior is guaranteed by fixing the random seed.
        """


        model = gpr.GaussianProcessBase(kernel=self.kernel, var_l=1.0, seed=0)
        sample = model.prior_sample(self.x_train)

        expected_sample = np.array([-3.57687542, -3.45727158, -2.72079989, -3.9645858,  -3.60895982])

        npt.assert_almost_equal(sample, expected_sample, decimal=7)

    def test_4(self):
        """
        Fit
        This test verifies that the regression base class _fit function behaves as
        expected.
        """

        model = gpr.GaussianProcessBase(kernel=self.kernel, var_l=[1, 2], seed=0)

        # var_l array is not the same size as the number of training observations
        with self.assertRaises(Exception) as context:
            model._fit(self.x_train, self.y_train)
        self.assertTrue("The length of the likelihood variance array does not match "
                        "the number of training observations." in str(context.exception))

        model = gpr.GaussianProcessBase(kernel=self.kernel, var_l=1, seed=0)
        K = model._fit(self.x_train, self.y_train)

        expected_K = np.array([[5.        , 3.61633139, 3.86140763, 3.73551499, 3.85672873],
                               [3.61633139, 5.        , 3.75073137, 3.84338362, 3.8831566 ],
                               [3.86140763, 3.75073137, 5.        , 3.83876279, 3.89740527],
                               [3.73551499, 3.84338362, 3.83876279, 5.        , 3.95976309],
                               [3.85672873, 3.8831566 , 3.89740527, 3.95976309, 5.        ]])

        npt.assert_almost_equal(K, expected_K, decimal=7)

        model = gpr.GaussianProcessBase(kernel=self.kernel, var_l=[1, 2, 3, 4, 5], seed=0)
        K = model._fit(self.x_train, self.y_train)

        expected_K = np.array([[5.        , 3.61633139, 3.86140763, 3.73551499, 3.85672873],
                               [3.61633139, 6.        , 3.75073137, 3.84338362, 3.8831566 ],
                               [3.86140763, 3.75073137, 7.        , 3.83876279, 3.89740527],
                               [3.73551499, 3.84338362, 3.83876279, 8.        , 3.95976309],
                               [3.85672873, 3.8831566 , 3.89740527, 3.95976309, 9.        ]])

        npt.assert_almost_equal(K, expected_K, decimal=7)

    def test_5(self):
        """
        Posterior Sample
        This test verifies that the regression base class posterior sample function
        behaves as expected.
        """

        # call posterior sample without fitting
        with self.assertRaises(Exception) as context:
            model = gpr.GaussianProcessBase(kernel=self.kernel, var_l=[1, 2], seed=0)
            model.posterior_sample(self.x_train)
        self.assertTrue("Please fit the model before trying to make posterior predictions!" in str(context.exception))

        def mock_posterior_predict(x_test, return_cov):
            return np.ones(3).reshape(-1, 1), np.identity(3)


        model = gpr.GaussianProcessBase(kernel=self.kernel, var_l=[1, 2], seed=0)
        model.posterior_predict = mock_posterior_predict
        model.fitted = True

        output = model.posterior_sample(self.x_train)

        expected_output = np.array([2.76405235, 1.40015721, 1.97873798])

        npt.assert_almost_equal(output, expected_output, decimal=7)


# ---------------------------------------------------------------------------------------------------------------------
# Tests for SOGP Inversion Class
# ---------------------------------------------------------------------------------------------------------------------

class GaussianProcessInversionTestCase(RegressionTestCase):
    """Tests for Gaussian Process Inversion Class"""

    def test_1(self):
        """
        Parameter Assertion
        Verifiy that model will not make posterior predictions if it has not been fit.
        """

        # call posterior predict without fitting
        with self.assertRaises(Exception) as context:
            model = gpr.GaussianProcessInversion(kernel=self.kernel, var_l=[1, 2], seed=0)
            model.posterior_predict(self.x_train)
        self.assertTrue("Please fit the model before trying to make posterior predictions!" in str(context.exception))

    def test_2(self):
        """
        Integration Test
        Integration test for gaussian process regression with matrix inversion.
        """
        # instantiate model object
        model = gpr.GaussianProcessInversion(kernel=self.kernel, var_l=1, seed=0)

        # fit model
        model.fit(self.x_train, self.y_train)

        # make a posterior prediction with variance
        output_mean, output_var = model.posterior_predict(self.x_train, False)

        expected_mean = np.array([0.5511188, 0.43958111, 0.56493794, 0.50156108, 0.50844248]).reshape(-1, 1)
        expected_var = np.array([0.35431073, 0.35068799, 0.29165543, 0.281214, 0.21375234]).reshape(-1, 1)

        npt.assert_almost_equal(output_mean, expected_mean, decimal=7)
        npt.assert_almost_equal(output_var, expected_var, decimal=7)

        # make a posterior prediction with covariance
        output_mean, output_cov = model.posterior_predict(self.x_train, True)

        expected_mean = np.array([0.5511188, 0.43958111, 0.56493794, 0.50156108, 0.50844248]).reshape(-1, 1)
        expected_cov = np.array([[0.35431073, 0.08071637, 0.20577217, 0.11965421, 0.18020684],
                                 [0.08071637, 0.35068799, 0.13381336, 0.1840639 , 0.19194095],
                                 [0.20577217, 0.13381336, 0.29165543, 0.15550502, 0.16634343],
                                 [0.11965421, 0.1840639 , 0.15550502, 0.281214  , 0.21278669],
                                 [0.18020684, 0.19194095, 0.16634343, 0.21278669, 0.21375234]])

        npt.assert_almost_equal(output_mean, expected_mean, decimal=7)
        npt.assert_almost_equal(output_cov, expected_cov, decimal=7)

# ---------------------------------------------------------------------------------------------------------------------
# Tests for SOGP Inversion Class
# ---------------------------------------------------------------------------------------------------------------------

class GaussianProcessCholeskyTestCase(RegressionTestCase):
    """Tests for Gaussian Process Inversion Class"""

    def test_1(self):
        """
        Parameter Assertion
        Verifiy that model will not make posterior predictions if it has not been fit.
        """

        # call posterior predict without fitting
        with self.assertRaises(Exception) as context:
            model = gpr.GaussianProcessCholesky(kernel=self.kernel, var_l=[1, 2], seed=0)
            model.posterior_predict(self.x_train)
        self.assertTrue("Please fit the model before trying to make posterior predictions!" in str(context.exception))

    def test_2(self):
        """
        Integration Test
        Integration test for gaussian process regression with matrix inversion.
        """
        # instantiate model object
        model = gpr.GaussianProcessCholesky(kernel=self.kernel, var_l=1, seed=0)

        # fit model
        model.fit(self.x_train, self.y_train)

        # make a posterior prediction with variance
        output_mean, output_var = model.posterior_predict(self.x_train, False)

        expected_mean = np.array([0.5511188, 0.43958111, 0.56493794, 0.50156108, 0.50844248]).reshape(-1, 1)
        expected_var = np.array([0.35431073, 0.35068799, 0.29165543, 0.281214, 0.21375234]).reshape(-1, 1)

        npt.assert_almost_equal(output_mean, expected_mean, decimal=7)
        npt.assert_almost_equal(output_var, expected_var, decimal=7)

        # make a posterior prediction with covariance
        output_mean, output_cov = model.posterior_predict(self.x_train, True)

        expected_mean = np.array([0.5511188, 0.43958111, 0.56493794, 0.50156108, 0.50844248]).reshape(-1, 1)
        expected_cov = np.array([[0.35431073, 0.08071637, 0.20577217, 0.11965421, 0.18020684],
                                 [0.08071637, 0.35068799, 0.13381336, 0.1840639 , 0.19194095],
                                 [0.20577217, 0.13381336, 0.29165543, 0.15550502, 0.16634343],
                                 [0.11965421, 0.1840639 , 0.15550502, 0.281214  , 0.21278669],
                                 [0.18020684, 0.19194095, 0.16634343, 0.21278669, 0.21375234]])

        npt.assert_almost_equal(output_mean, expected_mean, decimal=7)
        npt.assert_almost_equal(output_cov, expected_cov, decimal=7)


if __name__ == '__main__':
    unittest.main()
