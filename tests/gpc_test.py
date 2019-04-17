import unittest
import numpy as np
import numpy.testing as npt

from squidward import gpc, utils
from squidward.kernels import distance, kernel_base

# useful for debugging
np.set_printoptions(suppress=True)
np.random.seed(0)


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
        d = distance.RBF(5.0,1.0**2)
        # create kernel
        kernel = kernel_base.Kernel(d, 'k1')

        self.x_train = x_train
        self.y_train = y_train
        self.kernel = kernel


class GaussianProcessTestCase(ClassificationTestCase):
    """Tests for guassian process."""

    def test_1(self):
        """
        Prior Predict
        Prior predict should return not implemented.
        The prior over a one vs all gaussian process has no practical interpretation.
        """

        x_train = self.x_train
        y_train = self.y_train
        kernel = self.kernel

        # define model
        model = gpc.GaussianProcess(n_classes=3, kernel=kernel, var_l=1050**2)

        # check prior predict
        with self.assertRaises(NotImplementedError):
            model.prior_predict(x_train)

    def test_2(self):
        """
        Prior Sample Not Fit
        Test that the samples from the gpc prior before fitting make sense.
        """

        x_train = self.x_train
        y_train = self.y_train
        kernel = self.kernel

        # define model
        model = gpc.GaussianProcess(n_classes=7, kernel=kernel, var_l=0.1**2)

        # check prior sample without fitting
        logits = model.prior_sample(x_train, True)
        true = np.array([[-1.79067635,  1.09305832,  0.06399098, -0.14251713,  2.65510555,  1.35326968, -0.02482461],
                         [-1.75757787,  0.83630506, -0.27627176, -0.52926018,  2.4971101 ,  1.40032055, -0.24361749],
                         [-1.52572886,  1.01556157, -0.04330489, -0.22245361,  2.52040171,  1.58243624, -0.31409856],
                         [-1.89730377,  0.93145475, -0.28865612, -0.38382497,  2.4111784 ,  1.43099233, -0.0810563 ],
                         [-1.79204813,  0.97847252, -0.17067302, -0.3792619 ,  2.59903207,  1.45715448, -0.10615448]])

        npt.assert_almost_equal(logits.round(5), true.round(5), decimal=5)

        pred = model.prior_sample(x_train)
        true = np.array([[0.12480529, 0.1621448 , 0.14280534, 0.17604659, 0.09865555, 0.16052879, 0.13501364],
                         [0.10440005, 0.17551207, 0.15464492, 0.16258954, 0.0939536 , 0.15516955, 0.15373026],
                         [0.11712154, 0.18670788, 0.13810118, 0.1712029 , 0.09498927, 0.15366876, 0.13820848],
                         [0.10031737, 0.18469903, 0.14504828, 0.16995779, 0.09271834, 0.16213637, 0.14512282],
                         [0.10744391, 0.17595646, 0.14663246, 0.17046057, 0.09386693, 0.16097019, 0.14466948]])

        npt.assert_almost_equal(pred, true, decimal=5)

        prob_sums = pred.sum(axis=1)
        true = np.ones(5)
        npt.assert_almost_equal(prob_sums, true, decimal=5)

    def test_3(self):
        """
        Prior Sample Fit
        Test that the samples from the gpc prior after fitting make sense.
        """

        x_train = self.x_train
        y_train = self.y_train
        kernel = self.kernel

        # define model
        model = gpc.GaussianProcess(n_classes=7, kernel=kernel, var_l=1050**2)

        # fit model
        model.fit(x_train, y_train)

        # check prior sample after fitting
        logits = model.prior_sample(x_train, True)
        true = np.array([[-0.61679705,  0.55991878,  1.35042501, -1.67640677,  0.51837072, -0.66095565, -2.11626318],
                         [-0.65219386,  0.74951215,  1.07605748, -2.03375469,  0.25020942, -0.58177352, -1.78290087],
                         [-0.81536019,  0.59947552,  1.08057589, -1.91333909,  0.55759789, -0.67482517, -1.87641489],
                         [-0.82348693,  0.77639965,  1.06619781, -1.93809934,  0.30898196, -0.85869452, -1.6901788 ],
                         [-0.71318076,  0.71613839,  1.21532327, -1.85642939,  0.3678186 , -0.73280298, -1.89014316]])

        npt.assert_almost_equal(logits, true, decimal=5)

        pred = model.prior_sample(x_train)
        true = np.array([[0.04333675, 0.05466259, 0.13162566, 0.14371535, 0.2093348 , 0.27678021, 0.14054465],
                         [0.049477  , 0.04579106, 0.11176354, 0.17622607, 0.22958224, 0.26035738, 0.12680271],
                         [0.05382519, 0.06096332, 0.12594534, 0.15892353, 0.20749928, 0.25980799, 0.13303535],
                         [0.0525714 , 0.05436961, 0.11911843, 0.16516277, 0.19969524, 0.27478564, 0.1342969 ],
                         [0.05071181, 0.05142889, 0.12369409, 0.15913456, 0.21148355, 0.27230541, 0.13124169]])

        npt.assert_almost_equal(pred, true, decimal=5)

        prob_sums = true.sum(axis=1)
        true = np.ones(5)
        npt.assert_almost_equal(prob_sums, true, decimal=5)

    def test_4(self):
        """
        Posterior Predict
        Test that the statistics of the posterior of the gpc make sense.
        """

        x_train = self.x_train
        y_train = self.y_train
        kernel = self.kernel

        # define model
        model = gpc.GaussianProcess(n_classes=6, kernel=kernel, var_l=1050**2)

        # fit model
        model.fit(x_train, y_train)

        # check posterior predict
        logits_mean, logits_var = model.posterior_predict(x_train, True)
        true = np.array([[-0.00000087, -0.00000271, -0.00000088, -0.00000446, -0.00000446, -0.00000446],
                         [-0.00000092, -0.00000265, -0.00000089, -0.00000446, -0.00000446, -0.00000446],
                         [-0.0000009 , -0.00000271, -0.00000087, -0.00000448, -0.00000448, -0.00000448],
                         [-0.0000009 , -0.0000027 , -0.00000089, -0.00000448, -0.00000448, -0.00000448],
                         [-0.0000009 , -0.00000271, -0.00000089, -0.0000045 , -0.0000045 , -0.0000045 ]])

        npt.assert_almost_equal(logits_mean, true, decimal=5)

        true = np.array([[0.99999562, 0.99999562, 0.99999562, 0.99999562, 0.99999562, 0.99999562],
                         [0.99999561, 0.99999561, 0.99999561, 0.99999561, 0.99999561, 0.99999561],
                         [0.99999557, 0.99999557, 0.99999557, 0.99999557, 0.99999557, 0.99999557],
                         [0.99999557, 0.99999557, 0.99999557, 0.99999557, 0.99999557, 0.99999557],
                         [0.99999553, 0.99999553, 0.99999553, 0.99999553, 0.99999553, 0.99999553]])

        npt.assert_almost_equal(logits_var, true, decimal=5)

        pred = model.posterior_predict(x_train)
        true = np.array([[0.16666684, 0.16666669, 0.16666684, 0.16666654, 0.16666654, 0.16666654],
                         [0.16666684, 0.16666669, 0.16666684, 0.16666654, 0.16666654, 0.16666654],
                         [0.16666684, 0.16666669, 0.16666684, 0.16666654, 0.16666654, 0.16666654],
                         [0.16666684, 0.16666669, 0.16666684, 0.16666654, 0.16666654, 0.16666654],
                         [0.16666684, 0.16666669, 0.16666684, 0.16666654, 0.16666654, 0.16666654]])

        npt.assert_almost_equal(pred, true, decimal=5)

    def test_5(self):
        """
        Posterior Sample
        Test that the predictions from the gpc posterior sense.
        """

        x_train = self.x_train
        y_train = self.y_train
        kernel = self.kernel

        # define model
        model = gpc.GaussianProcess(n_classes=3, kernel=kernel, var_l=1050**2)

        # fit model
        model.fit(x_train, y_train)

        # check posterior sample
        logits = model.posterior_sample(x_train, True)
        true = np.array([[ 1.48918898, -0.96043104,  0.3211009 ],
                         [ 1.41376419, -0.73081545, -0.19096082],
                         [ 1.56668453, -1.09638425,  0.08575354],
                         [ 1.41274437, -0.97055041,  0.04850063],
                         [ 1.5253572 , -0.93287597,  0.0746793 ]])
        
        npt.assert_almost_equal(logits, true, decimal=5)

        pred = model.posterior_sample(x_train)
        true = np.array([[0.4259497 , 0.41844162, 0.15560868],
                         [0.40883743, 0.38773097, 0.20343161],
                         [0.40351694, 0.40441486, 0.19206819],
                         [0.42116111, 0.36615497, 0.21268391],
                         [0.42222255, 0.38345229, 0.19432516]])

        npt.assert_almost_equal(pred, true, decimal=5)

    def test_6(self):
        """
        Params Assertions
        Test that the gpc assertions work to raise exceptions for invalid parameters.
        """

        x_train = self.x_train
        y_train = self.y_train
        kernel = self.kernel

        # define model
        model = gpc.GaussianProcess(n_classes=2, kernel=kernel, var_l=1050**2)

        with self.assertRaises(Exception) as context:
            model.fit(x_train, y_train)
        self.assertTrue('More classes in ytrain than specified in model object.' in str(context.exception))

        with self.assertRaises(Exception) as context:
            gpc.GaussianProcess(var_l=1050**2)
        self.assertTrue('Please specify the number of classes.' in str(context.exception))

        with self.assertRaises(Exception) as context:
            gpc.GaussianProcess(n_classes=5, var_l=1050**2)
        self.assertTrue('Model object must be instantiated with a valid kernel object.' in str(context.exception))

        with self.assertRaises(Exception) as context:
            gpc.GaussianProcess(n_classes=5, kernel=kernel, var_l=-1.0**2)
        self.assertTrue('Invalid likelihood variance argument.' in str(context.exception))

        with self.assertRaises(Exception) as context:
            gpc.GaussianProcess(n_classes=5, kernel=kernel, var_l=1050**2).posterior_predict(x_train)
        self.assertTrue("Please fit the model before trying to make posterior predictions!" in str(context.exception))

        with self.assertRaises(Exception) as context:
            gpc.GaussianProcess(n_classes=5, kernel=kernel, var_l=1050**2).posterior_sample(x_train)
        self.assertTrue("Please fit the model before trying to make posterior predictions!" in str(context.exception))


if __name__ == '__main__':
    unittest.main(failfast=True)
