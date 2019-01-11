import unittest
import numpy as np
from squidward import GPR
from squidward.Kernels import Distance, KernelBase
import numpy.testing as npt
np.random.seed(0)

# useful for debugging
np.set_printoptions(suppress=True)

class RegressionTestCase(unittest.TestCase):
    """
    Class for guassian process classification tests.
    """
    def setUp(self):
        """
        Set up shared environment or variables for tests.
        """
        # train data
        x_train = np.array([[[0.52954542, 0.60434688, 0.92717256, 0.69505128, 0.21809012],
                            [0.5179508 , 0.88071251, 0.78696724, 0.28974399, 0.51559588],
                            [0.90394506, 0.15472882, 0.26058205, 0.73792192, 0.50366436],
                            [0.2180652 , 0.73934024, 0.4229633 , 0.20175262, 0.83121799]],
                           [[0.4682661 , 0.87468674, 0.90483934, 0.42801402, 0.7750081 ],
                            [0.22594395, 0.00606466, 0.28807989, 0.02169302, 0.52709627],
                            [0.20366729, 0.92742153, 0.23589016, 0.30158137, 0.79468193],
                            [0.17676099, 0.69873352, 0.05348105, 0.5578392 , 0.438718  ]],
                           [[0.23976736, 0.85409227, 0.27421305, 0.57186676, 0.34302778],
                            [0.26049829, 0.10676582, 0.63218182, 0.17066523, 0.53458649],
                            [0.24315636, 0.8612443 , 0.32608555, 0.46411391, 0.27134582],
                            [0.93267862, 0.36709855, 0.41112865, 0.91878905, 0.021376  ]]])
        y_train = np.array([0.63095398, 0.29118303, 0.85576359])

        # create distance function
        d = Distance.rbf(5.0,10000.0**2)
        # create kernel
        kernel = KernelBase.Kernel(d, 'k1')

        self.x_train = x_train
        self.y_train = y_train
        self.kernel = kernel

class GaussianProcessTestCase(RegressionTestCase):
    """
    Tests for guassian process.
    """

    def test_prior_prediction(self):
        """
        Test that the prior statistics for GPR makes sense.
        """
        x_train = self.x_train
        y_train = self.y_train
        kernel = self.kernel

        # define model
        model = GPR.GaussianProcess(kernel=kernel, var_l=1050**2, inv_method='lu')

        # check prior predict
        mean, var = model.prior_predict(x_train)
        true = np.zeros(3).reshape(-1,1)
        npt.assert_almost_equal(mean, true, decimal=10)
        true = np.full((3,1), 1e8)
        npt.assert_almost_equal(var, true, decimal=10)

        mean, cov = model.prior_predict(x_train, True)
        true = np.array([[1.          , 0.934002353 , 0.9195110831],
                         [0.934002353 ,1.          , 0.9562317048],
                         [0.9195110831, 0.9562317048, 1.          ]])
        cov /= 1e8
        npt.assert_almost_equal(cov, true, decimal=10)

    def test_prior_sample(self):
        """
        Test that samples from the prior make sense.
        """
        x_train = self.x_train
        y_train = self.y_train
        kernel = self.kernel

        # define model
        model = GPR.GaussianProcess(kernel=kernel, var_l=1050**2, inv_method='lu')

        # check prior sample
        sample = []
        for i in range(100):
            sample.append( model.prior_sample(x_train) )
        mean = np.mean(sample, axis=0)
        true = np.array([-90.9387249999, 820.9311221566, 336.3775325309])
        npt.assert_almost_equal(mean, true, decimal=10)
        var = np.std(sample, axis=0)**2
        true = np.array([1.0268415343, 1.0282576537, 0.9955303171])
        var /= 1e8
        npt.assert_almost_equal(var, true, decimal=10)

    def test_posterior_prediction(self):
        """
        Test that posterior statistics make sense.
        """
        x_train = self.x_train
        y_train = self.y_train
        kernel = self.kernel

        # define model
        model = GPR.GaussianProcess(kernel=kernel, var_l=1050**2, inv_method='lu')

        # fit model
        model.fit(self.x_train, self.y_train)

        # check posterior predict
        mean, var = model.posterior_predict(x_train)
        true = np.array([0.6175477307, 0.3528573286, 0.8002926388]).reshape(-1,1)
        npt.assert_almost_equal(mean, true, decimal=10)
        true = np.array([1013082.0902103633,  955944.8353241384,  978226.0181837529]).reshape(-1,1)
        npt.assert_almost_equal(var, true, decimal=10)

        mean, cov = model.posterior_predict(x_train, True)
        true = np.array([[1013082.0902103633,   53948.2113315463,   30299.714634344 ],
                         [53948.211331591 ,  955944.8353241384,   89547.4560313374],
                         [30299.7146342993,   89547.4560312331,  978226.0181837529]])
        npt.assert_almost_equal(cov, true, decimal=10)

    def test_posterior_sample(self):
        """
        Test that samples from the posterior make sense.
        """
        x_train = self.x_train
        y_train = self.y_train
        kernel = self.kernel

        # define model
        model = GPR.GaussianProcess(kernel=kernel, var_l=1050**2, inv_method='lu')

        # fit model
        model.fit(self.x_train, self.y_train)

        # check posterior sample
        sample = []
        for i in range(100):
            sample.append( model.posterior_sample(x_train) )
        mean = np.mean(sample, axis=0)
        true = np.array([   5.2083344224,    2.0672928144, -187.7642256326])
        npt.assert_almost_equal(mean, true, decimal=10)
        var = np.std(sample, axis=0)**2
        true = np.array([1053749.1595542012, 1015560.0043658405,  859673.284072937 ])
        npt.assert_almost_equal(var, true, decimal=10)

    def test_params_assertions(self):
        """
        Test that the GPR assertions work to raise exceptions for invalid parameters.
        """
        kernel = self.kernel
        x_train = self.x_train

        with self.assertRaises(Exception) as context:
            GPR.GaussianProcess(var_l=1050**2, inv_method='lu')
        self.assertTrue('Model object must be instantiated with a valid kernel object.' in str(context.exception))

        with self.assertRaises(Exception) as context:
            GPR.GaussianProcess(kernel=kernel, var_l=-1.0**2, inv_method='lu')
        self.assertTrue('Invalid likelihood variance argument.' in str(context.exception))

        with self.assertRaises(Exception) as context:
            GPR.GaussianProcess(kernel=kernel, var_l=1050**2, inv_method='lu').posterior_predict(x_train)
        self.assertTrue('Please fit the model before trying to make posterior predictions!' in str(context.exception))

class GaussianProcessStableCholeskyTestCase(RegressionTestCase):
    """
    Tests for guassian process stable cholesky.
    """

    def test_guassian_process_stable_cholesky(self):
        """
        Test that the stable cholesky implementation of GPR returns valid
        posterior predictions.
        """
        x_train = self.x_train
        y_train = self.y_train
        kernel = self.kernel

        model = GPR.GaussianProcessStableCholesky(kernel=kernel, var_l=1050**2)

        mean, var = model.fit_predict(x_train, y_train, x_train)
        true = np.array([0.6175477307, 0.3528573286, 0.8002926388]).reshape(-1,1)
        npt.assert_almost_equal(mean, true, decimal=10)
        true = np.array([1013082.0902104378,  955944.8353241831,  978226.018183887 ]).reshape(-1,1)
        npt.assert_almost_equal(var, true, decimal=10)

        mean, cov = model.fit_predict(x_train, y_train, x_train, True)
        true = np.array([[1013082.0902104378,   53948.2113316208,   30299.7146344185],
                         [53948.2113316208,  955944.8353241831,   89547.4560313672],
                         [30299.7146344185,   89547.4560313672,  978226.018183887 ]])
        npt.assert_almost_equal(cov, true, decimal=10)

    def test_params_assertions(self):
        """
        Test that the GPR assertions work to raise exceptions for invalid parameters.
        """
        kernel = self.kernel

        with self.assertRaises(Exception) as context:
            GPR.GaussianProcessStableCholesky(var_l=1050**2)
        self.assertTrue('Model object must be instantiated with a valid kernel object.' in str(context.exception))

        with self.assertRaises(Exception) as context:
            GPR.GaussianProcessStableCholesky(kernel=kernel, var_l=-1.0**2)
        self.assertTrue('Invalid likelihood variance argument.' in str(context.exception))

if __name__ == '__main__':
    unittest.main()
