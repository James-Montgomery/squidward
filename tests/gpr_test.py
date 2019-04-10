import unittest
import numpy as np
from squidward import gpr
from squidward.kernels import distance, kernel_base
import numpy.testing as npt
np.random.seed(0)

# useful for debugging
np.set_printoptions(suppress=True)

class RegressionTestCase(unittest.TestCase):
    """Class for guassian process classification tests."""

    def setUp(self):
        """
        Set Up
        Set up shared environment or variables for tests.
        """
        # train data
        x_train = np.array([[0.74775167, 0.09963786, 0.24078391,
                           0.16182137, 0.72764008, 0.13583729],
                          [0.1427068 , 0.61315893, 0.61929491,
                           0.88585838, 0.96463067, 0.81522606],
                          [0.16291094, 0.59166813, 0.17442026,
                           0.34366403, 0.75932132, 0.02574967],
                          [0.16449091, 0.71445737, 0.71023514,
                           0.22225375, 0.49399307, 0.62012281],
                          [0.36558453, 0.50613319, 0.62597648,
                          0.44873197, 0.5863159 , 0.44340289]])
        y_train = np.array([0.63095398, 0.29118303, 0.85576359,
                            0.63095398, 0.29118303])

        # create distance function
        d = distance.RBF(1.0,1.0**2)
        # create kernel
        kernel = kernel_base.Kernel(d, 'k1')

        self.x_train = x_train
        self.y_train = y_train
        self.kernel = kernel

class GaussianProcessTestCase(RegressionTestCase):
    """Tests for guassian process."""

    def test_1(self):
        """
        Prior Prediction
        Test that the prior statistics for gpr makes sense.
        """
        x_train = self.x_train
        y_train = self.y_train
        kernel = self.kernel

        # define model
        model = gpr.GaussianProcess(kernel=kernel, var_l=1.0**2, inv_method='lu')

        # check prior predict
        mean, var = model.prior_predict(x_train)
        true = np.zeros(5).reshape(-1,1)
        npt.assert_almost_equal(mean, true, decimal=10)
        true = np.full((5,1), 1)
        npt.assert_almost_equal(var, true, decimal=10)

        mean, cov = model.prior_predict(x_train, True)
        true = np.array([[1.        , 0.40352836, 0.7280663 , 0.54027445, 0.72016484],
                         [0.40352836, 1.        , 0.56040723, 0.69804543, 0.76581563],
                         [0.7280663 , 0.56040723, 1.        , 0.69052844, 0.79148045],
                         [0.54027445, 0.69804543, 0.69052844, 1.        , 0.91302549],
                         [0.72016484, 0.76581563, 0.79148045, 0.91302549, 1.        ]])
        npt.assert_almost_equal(cov, true, decimal=7)

    def test_2(self):
        """
        Prior Sample
        Test that samples from the prior make sense.
        """
        x_train = self.x_train
        y_train = self.y_train
        kernel = self.kernel

        # define model
        model = gpr.GaussianProcess(kernel=kernel, var_l=1050**2, inv_method='lu')

        # check prior sample
        sample = []
        for i in range(100):
            sample.append( model.prior_sample(x_train) )
        mean = np.mean(sample, axis=0)
        true = np.array([0.076381005 , 0.0627892362, 0.0462841433, 0.1107081955, 0.1407366603])
        npt.assert_almost_equal(mean, true, decimal=10)

        var = np.std(sample, axis=0)**2
        true = np.array([1.0356226631, 1.0394182763, 1.002835054 , 0.9181960839, 1.0255528133])
        npt.assert_almost_equal(var, true, decimal=10)

    def test_3(self):
        """
        Posterior Prediction
        Test that posterior statistics make sense.
        """
        x_train = self.x_train
        y_train = self.y_train
        kernel = self.kernel

        # define model
        model = gpr.GaussianProcess(kernel=kernel, var_l=1050**2, inv_method='lu')

        # fit model
        model.fit(self.x_train, self.y_train)

        # check posterior predict
        mean, var = model.posterior_predict(x_train)
        true = np.array([[0.00000174],
                         [0.00000153],
                         [0.00000195],
                         [0.00000184],
                         [0.00000202]])

        npt.assert_almost_equal(mean, true, decimal=7)

        true = np.array([[0.99999773],
                         [0.99999769],
                         [0.99999733],
                         [0.9999972 ],
                         [0.99999677]])

        npt.assert_almost_equal(var, true, decimal=7)

        mean, cov = model.posterior_predict(x_train, True)
        true = np.array([[0.99999773, 0.40352642, 0.72806391, 0.54027216, 0.72016228],
                         [0.40352642, 0.99999769, 0.56040496, 0.69804298, 0.76581299],
                         [0.72806391, 0.56040496, 0.99999733, 0.69052582, 0.79147758],
                         [0.54027216, 0.69804298, 0.69052582, 0.9999972 , 0.91302251],
                         [0.72016228, 0.76581299, 0.79147758, 0.91302251, 0.99999677]])

        npt.assert_almost_equal(cov, true, decimal=7)

    def test_4(self):
        """
        Posterior Sample
        Test that samples from the posterior make sense.
        """
        x_train = self.x_train
        y_train = self.y_train
        kernel = self.kernel

        # define model
        model = gpr.GaussianProcess(kernel=kernel, var_l=1050**2, inv_method='lu')

        # fit model
        model.fit(self.x_train, self.y_train)

        # check posterior sample
        sample = []
        for i in range(100):
            sample.append( model.posterior_sample(x_train) )
        mean = np.mean(sample, axis=0)
        true = np.array([0.02596258, 0.14195107, 0.04089343, 0.12286752, 0.09163197])
        
        npt.assert_almost_equal(mean, true, decimal=7)

        var = np.std(sample, axis=0)**2
        true = np.array([1.06269602, 0.92896852, 1.0481692,  1.0479876,  1.03666952])

        npt.assert_almost_equal(var, true, decimal=7)

    def test_5(self):
        """
        Params Assertions
        Test that the gpr assertions work to raise exceptions for invalid parameters.
        """
        kernel = self.kernel
        x_train = self.x_train

        with self.assertRaises(Exception) as context:
            gpr.GaussianProcess(var_l=1050**2, inv_method='lu')
        self.assertTrue('Model object must be instantiated with a valid kernel object.' in str(context.exception))

        with self.assertRaises(Exception) as context:
            gpr.GaussianProcess(kernel=kernel, var_l=-1.0**2, inv_method='lu')
        self.assertTrue('Invalid likelihood variance argument.' in str(context.exception))

        with self.assertRaises(Exception) as context:
            gpr.GaussianProcess(kernel=kernel, var_l=1050**2, inv_method='lu').posterior_predict(x_train)
        self.assertTrue('Please fit the model before trying to make posterior predictions!' in str(context.exception))

class GaussianProcessStableCholeskyTestCase(RegressionTestCase):
    """Tests for guassian process stable cholesky."""

    def test_1(self):
        """
        Guassian Process Stable Cholesky
        Test that the stable cholesky implementation of gpr returns valid
        posterior predictions.
        """
        x_train = self.x_train
        y_train = self.y_train
        kernel = self.kernel

        model = gpr.GaussianProcessStableCholesky(kernel=kernel, var_l=1050**2)

        mean, var = model.fit_predict(x_train, y_train, x_train)
        true = np.array([0.0000017434, 0.0000015318, 0.0000019451,
                         0.000001843 , 0.0000020154]).reshape(-1,1)
        npt.assert_almost_equal(mean, true, decimal=10)
        true = np.array([0.9999977293, 0.9999976865, 0.9999973266,
                         0.9999971976, 0.9999967663]).reshape(-1,1)
        npt.assert_almost_equal(var, true, decimal=10)

        mean, cov = model.fit_predict(x_train, y_train, x_train, True)
        true = np.array([[0.99999773, 0.40352642, 0.72806391, 0.54027216, 0.72016228],
                         [0.40352642, 0.99999769, 0.56040496, 0.69804298, 0.76581299],
                         [0.72806391, 0.56040496, 0.99999733, 0.69052582, 0.79147758],
                         [0.54027216, 0.69804298, 0.69052582, 0.9999972 , 0.91302251],
                         [0.72016228, 0.76581299, 0.79147758, 0.91302251, 0.99999677]])
        npt.assert_almost_equal(cov, true, decimal=7)

    def test_2(self):
        """
        Params Assertions
        Test that the gpr assertions work to raise exceptions for invalid parameters.
        """
        kernel = self.kernel

        with self.assertRaises(Exception) as context:
            gpr.GaussianProcessStableCholesky(var_l=1050**2)
        self.assertTrue('Model object must be instantiated with a valid kernel object.' in str(context.exception))

        with self.assertRaises(Exception) as context:
            gpr.GaussianProcessStableCholesky(kernel=kernel, var_l=-1.0**2)
        self.assertTrue('Invalid likelihood variance argument.' in str(context.exception))

if __name__ == '__main__':
    unittest.main()
