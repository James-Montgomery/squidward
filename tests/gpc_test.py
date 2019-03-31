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

        y_train = np.array([0,1,2,0,2])
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
        true = np.array([[ 6.79441613,  2.33601883, -6.16797048,  5.59918781, 13.50425015],
                         [ 6.67855488,  6.40610254, -6.5219386 ,  7.4951215 , 10.76057477],
                         [ 4.90378057,  2.32233898, -8.15360192,  5.99475517, 10.80575891],
                         [ 7.65826068,  4.5146641 , -8.2348693 ,  7.76399648, 10.66197811],
                         [ 7.36801364,  4.38118452, -7.13180755,  7.16138391, 12.15323272]])
        logits /= 1000
        npt.assert_almost_equal(logits.round(5), true.round(5), decimal=5)



        pred = model.prior_sample(x_train, 5)
        true = np.array([[0.16666667, 0.33333333, 0.16666667, 0.16666667, 0.16666667],
                         [0.16666667, 0.33333333, 0.16666667, 0.16666667, 0.16666667],
                         [0.16666667, 0.33333333, 0.16666667, 0.16666667, 0.16666667],
                         [0.16666667, 0.33333333, 0.16666667, 0.16666667, 0.16666667],
                         [0.16666667, 0.33333333, 0.16666667, 0.16666667, 0.16666667]])
        npt.assert_almost_equal(pred, true, decimal=5)

        prob_sums = pred.sum(axis=1)
        true = np.ones(5)
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
        true = np.array([[ -248.24608649,   637.64724596,  7095.17587862],
                         [-2436.17487503, -2219.28149111, 10862.87661892],
                         [-3140.98562788, -1093.34152028, 11175.76189559],
                         [ -810.56295113, -3131.31822664, 12517.88721622],
                         [-1061.54478592, -1949.66948799, 10421.20316781]])
        npt.assert_almost_equal(logits, true, decimal=5)

        pred = model.prior_sample(x_train, None)
        true = np.array([[0.4, 0.4, 0.2],
                         [0.4, 0.4, 0.2],
                         [0.4, 0.4, 0.2],
                         [0.4, 0.4, 0.2],
                         [0.4, 0.4, 0.2]])
        npt.assert_almost_equal(pred, true, decimal=5)

        prob_sums = true.sum(axis=1)
        true = np.ones(5)
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
        true = np.array([[ 0.39533686, -1.07071283, -0.31980672],
                         [-0.88143911,  0.42095943, -0.53481576],
                         [-0.51579997, -0.8921191 ,  0.40949056],
                         [ 0.11798784, -0.80783398, -0.30737432],
                         [-0.12008728, -0.64851057, -0.23408299]])
        npt.assert_almost_equal(logits_mean, true, decimal=5)
        true = np.array([[770959.82101963, 770959.82101963, 770959.82101963],
                         [780710.496435  , 780710.496435  , 780710.496435  ],
                         [680817.79456477, 680817.79456477, 680817.79456477],
                         [619194.49772668, 619194.49772668, 619194.49772668],
                         [328391.97284918, 328391.97284918, 328391.97284918]])
        npt.assert_almost_equal(logits_var, true, decimal=5)

        pred = model.posterior_predict(x_train)
        true = np.array([[0.4692108 , 0.20043666, 0.33035254],
                         [0.23134494, 0.47687131, 0.29178375],
                         [0.2954111 , 0.22969486, 0.47489403],
                         [0.41968535, 0.2444193 , 0.33589535],
                         [0.3744881 , 0.27354753, 0.35196437]])
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
        true = np.array([[-17906.7635222 ,  10930.58323709,    639.90977875],
                         [-17575.77871002,   8363.05058755,  -2762.7176007 ],
                         [-15257.2885911 ,  10155.61567795,   -433.04889641],
                         [-18973.03766274,   9314.54749922,  -2886.5611732 ],
                         [-17920.48129187,   9784.72516109,  -1706.73015691]])
        npt.assert_almost_equal(logits, true, decimal=5)

        pred = model.posterior_sample(x_train)
        true = np.array([[0.2, 0.4, 0.4],
                         [0.2, 0.4, 0.4],
                         [0.2, 0.4, 0.4],
                         [0.2, 0.4, 0.4],
                         [0.2, 0.4, 0.4]])
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
