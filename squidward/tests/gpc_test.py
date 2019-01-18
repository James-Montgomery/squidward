import unittest
import numpy as np
from squidward import gpc, utils
from squidward.kernels import distance, kernel_base
import numpy.testing as npt
np.random.seed(0)

# useful for debugging
np.set_printoptions(suppress=True)

class ClassificationTestCase(unittest.TestCase):
    """Class for guassian process classification tests."""

    def setUp(self):
        """
        Set Up
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

        y_train = np.array([0,1,2])
        y_train = utils.onehot(y_train)

        # create distance function
        d = distance.RBF(5.0,10000.0**2)
        # create kernel
        kernel = kernel_base.Kernel(d, 'k1')

        self.x_train = x_train
        self.y_train = y_train
        self.kernel = kernel

class GaussianProcessTestCase(ClassificationTestCase):
    """Tests for guassian process."""

    def test_prior_predict(self):
        """
        Prior Predict
        Prior predict should return not implemented.
        The prior over a one vs all gaussian process has no practical interpretation.
        """
        x_train = self.x_train
        y_train = self.y_train
        kernel = self.kernel

        # define model
        model = gpc.GaussianProcess(kernel=kernel, var_l=1050**2, inv_method='lu')

        # check prior predict
        with self.assertRaises(NotImplementedError):
            model.prior_predict(x_train)


    def test_prior_sample_not_fit(self):
        """
        Prior Sample Not Fit
        Test that the samples from the gpc prior before fitting make sense.
        """
        x_train = self.x_train
        y_train = self.y_train
        kernel = self.kernel

        # define model
        model = gpc.GaussianProcess(kernel=kernel, var_l=1050**2, inv_method='lu')

        # check prior sample without fitting
        logits = model.prior_sample(x_train, 5, True)
        true = np.array([[ -9.2786 ,   0.00624,  20.9638 ,   1.59503,  14.88197],
                         [-12.39296, 6.00605,  16.15685,   4.01228,  17.46699],
                         [-14.42566,   2.82415, 13.012  ,   7.23432,  15.01898]])
        logits /= 1000
        npt.assert_almost_equal(logits.round(5), true.round(5), decimal=5)



        pred = model.prior_sample(x_train, 5)
        true = np.array([[0.125  , 0.25   , 0.125  , 0.25   , 0.25   ],
                         [0.14286, 0.14286, 0.14286, 0.28571, 0.28571],
                         [0.14286, 0.14286, 0.14286, 0.28571, 0.28571]])
        npt.assert_almost_equal(pred, true, decimal=5)

        prob_sums = pred.sum(axis=1)
        true = np.ones(3)
        npt.assert_almost_equal(prob_sums, true, decimal=5)

    def test_prior_sample_fit(self):
        """
        Prior Sample Fit
        Test that the samples from the gpc prior after fitting make sense.
        """
        x_train = self.x_train
        y_train = self.y_train
        kernel = self.kernel

        # define model
        model = gpc.GaussianProcess(kernel=kernel, var_l=1050**2, inv_method='lu')

        # fit model
        model.fit(x_train, y_train)

        # check prior sample after fitting
        logits = model.prior_sample(x_train, None, True)
        true = np.array([[ -5904.28677,  -4584.64068, -25437.60271],
                         [1598.20037, -5901.75148, -21338.57819],
                         [-4931.93158,  -8691.74403,-19893.62425]])
        npt.assert_almost_equal(logits, true, decimal=5)

        pred = model.prior_sample(x_train, None)
        true = np.array([[0.4 , 0.2 , 0.4 ],
                         [0.25, 0.25, 0.5 ],
                         [0.4 , 0.2 , 0.4 ]])
        npt.assert_almost_equal(pred, true, decimal=5)

        prob_sums = true.sum(axis=1)
        true = np.ones(3)
        npt.assert_almost_equal(prob_sums, true, decimal=5)

    def test_posterior_predict(self):
        """
        Posterior Predict
        Test that the statistics of the posterior of the gpc make sense.
        """
        x_train = self.x_train
        y_train = self.y_train
        kernel = self.kernel

        # define model
        model = gpc.GaussianProcess(kernel=kernel, var_l=1050**2, inv_method='lu')

        # fit model
        model.fit(x_train, y_train)

        # check posterior predict
        logits_mean, logits_var = model.posterior_predict(x_train, True)
        true = np.array([[ 0.84248, -0.89745, -0.94035],
                         [-0.89936,  0.73692, -0.83478],
                         [-0.94102, -0.83354,  0.77857]])
        npt.assert_almost_equal(logits_mean, true, decimal=5)
        true = np.array([[1013082.09021, 1013082.09021, 1013082.09021],
                         [955944.83532, 955944.83532,  955944.83532],
                         [978226.01818,  978226.01818, 978226.01818]])
        npt.assert_almost_equal(logits_var, true, decimal=5)

        pred = model.posterior_predict(x_train)
        true = np.array([[0.55065, 0.22812, 0.22123],
                         [0.22804, 0.53332, 0.23865],
                         [0.2212 , 0.2387 , 0.5401 ]])
        npt.assert_almost_equal(pred, true, decimal=5)


    def test_posterior_sample(self):
        """
        Posterior Sample
        Test that the predictions from the gpc posterior sense.
        """
        x_train = self.x_train
        y_train = self.y_train
        kernel = self.kernel

        # define model
        model = gpc.GaussianProcess(kernel=kernel, var_l=1050**2, inv_method='lu')

        # fit model
        model.fit(x_train, y_train)

        # check posterior sample
        logits = model.posterior_sample(x_train, True)
        true = np.array([[-15875.07385, -17745.91421,  -9621.04409],
                         [-19216.62584, -21902.53037,  -9078.1328 ],
                         [ -16679.511, -26110.41987, -9198.9402 ]])
        npt.assert_almost_equal(logits, true, decimal=5)

        pred = model.posterior_sample(x_train)
        true = np.array([[0.25   , 0.25   , 0.5    ],
                         [0.33333, 0.33333, 0.33333],
                         [0.33333, 0.33333, 0.33333]])
        npt.assert_almost_equal(pred, true, decimal=5)

    def test_params_assertions(self):
        """
        Params Assertions
        Test that the gpc assertions work to raise exceptions for invalid parameters.
        """
        x_train = self.x_train
        kernel = self.kernel

        with self.assertRaises(Exception) as context:
            gpc.GaussianProcess(var_l=1050**2, inv_method='lu')
        self.assertTrue('Model object must be instantiated with a valid kernel object.' in str(context.exception))

        with self.assertRaises(Exception) as context:
            gpc.GaussianProcess(kernel=kernel, var_l=-1.0**2, inv_method='lu')
        self.assertTrue('Invalid likelihood variance argument.' in str(context.exception))

        with self.assertRaises(Exception) as context:
            gpc.GaussianProcess(kernel=kernel, var_l=1050**2, inv_method='lu').posterior_predict(x_train)
        self.assertTrue('Please fit the model before trying to make posterior predictions!' in str(context.exception))

        with self.assertRaises(Exception) as context:
            model = gpc.GaussianProcess(kernel=kernel, var_l=1050**2, inv_method='lu')
            model.prior_sample(x_train)
        self.assertTrue('Please either fit the model or specify the number of classes.' in str(context.exception))

if __name__ == '__main__':
    unittest.main()
