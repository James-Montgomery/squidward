"""
This script contains code for basic gaussian process regression. A regression
model can be created by calling one of these classes to create a model object.
"""

import numpy as np
from squidward.utils import Invert, exactly_1d, exactly_2d, check_valid_cov

# TODO: Add tests for non-zero prior mean
# TODO: set up logging options
# TODO: bigfloat
# https://stackoverflow.com/questions/9559346/deal-with-overflow-in-exp-using-numpy/9559478

np.seterr(over="raise")

# ---------------------------------------------------------------------------------------------------------------------
# Gaussian Process Regression
# Base Class
# ---------------------------------------------------------------------------------------------------------------------


class GaussianProcessBase(object):
    """Base Class Gaussian Process Regression"""

    def __init__(self, kernel=None, prior_mean=None, var_l=1e-15, seed=None, show_warnings=True):
        """
        Description
        ----------
        Base class foe gaussian process regression model object.

        Parameters
        ----------
        kernel : kernel object
            An object with an associated function k that takes in 2 arrays and
            returns a valid K matrix. Valid K matricies are positive
            semi-definite and not singular.
        prior_mean: function
            The mean of the GP prior. Defaults to a zero mean GP. Should be a function
            that takes in a features array x and returns a prediction array y. The output
            should be an (n, 1) numpy array.
        var_l: float or array_like
            The likelihood variance of the process. Takes a scalar for homoscedastic
            regression and an array_like for heteroscedastic regression. If an array,
            the variances should be ordered with respect to the training observations.
        seed: integer
            An optional parameter that sets the seed for prior and posterior
            sampling functions.
        show_warnings: boolean
            A boolean indicating whether to silence singular matrix warnings.
            Defaults to show warnings (better safe than sorry).

        Returns
        ----------
        Model object
        """
        self.safe = show_warnings

        self.kernel = kernel
        self.var_l = var_l

        # Gaussian Processes for Machine Learning Eq 2.7
        self.prior_mean = prior_mean

        self.x_obs = None
        self.y_obs = None

        self.K = None
        self.fitted = False

        # Assertions to ensure that inputs are of expected types

        assert (self.prior_mean is None) or callable(self.prior_mean), \
            "Invalid prior mean function."

        assert isinstance(self.var_l, (int, float, list, np.ndarray)), \
            "Likelihood variance argument must be a positive integer, " \
            "float, array of integers, or array of floats."

        # homoscedastic case
        if isinstance(self.var_l, (int,float)):
            assert self.var_l >= 0.0, \
                "Likelihood variance argument must be a positive integer, " \
                "float, array of integers, or array of floats."
        # heteroscedastic case
        else:
            try:
                self.var_l = exactly_1d( np.asarray(self.var_l) )
                assert self.var_l.shape[0] > 0, \
                    "Likelihood variance argument must be a positive integer, " \
                    "float, array of integers, or array of floats."
            except Exception as e:
                error = "Error with likelihood variance argument.\n{}".format(e)
                raise Exception(error)

            assert (self.var_l >= 0.0).all(), \
                "Likelihood variance argument must be a positive integer, " \
                "float, array of integers, or array of floats."

        assert (seed is None) or isinstance(seed, int), \
            "When specifying random seed argument it must be an integer."

        assert isinstance(self.safe, bool), \
            "Show warnings argument must be a boolean."

        assert self.kernel is not None, \
            "Model object must be instantiated with a valid kernel object."

        if seed is None:
            self.random = np.random
        else:
            self.random = np.random.RandomState(seed)

    def prior_predict(self, x_test, return_cov=False):
        """
        Description
        ----------
        Make predictions. This function takes in a set of test points to make
        predictions on and returns the mean function of the prior of the
        gaussian process and a measure of uncertainty (either covariance or
        variance).

        Parameters
        ----------
        x_test: array_like
            Feature input for points to make predictions for.
        return_cov: boolean
            If true, will return the full covariance matrix. Otherwise it will
            return the variance.

        Returns
        ----------
        Mean: array_like
            An array with the values of the mean function of the guassian
            process prior.
        Var: array_like
            The variance around the values of the mean function of the
            gaussian process posterior. If return_cov is True then returns the
            full covariance matrix of the gaussian process posterior instead.
        """

        if self.prior_mean is None:
            mean = np.zeros(x_test.shape[0]).reshape(-1, 1)
        else:
            mean = exactly_1d(self.prior_mean(self.x_obs))
        mean = exactly_2d(mean)

        cov = self.kernel(x_test, x_test)

        check_valid_cov(cov, self.safe)
        if return_cov:
            return mean, cov

        var = exactly_2d(np.diag(cov))
        return mean, var

    def prior_sample(self, x_test):
        """
        Description
        ----------
        Draw a function from the prior.

        Parameters
        ----------
        x_test: array_like
            Feature input for points to draw samples for.

        Returns
        ----------
        Sample: array_like
            The values of a function sampled from the gaussian process prior.
        """
        mean, cov = self.prior_predict(x_test, True)
        return self.random.multivariate_normal(mean[:, 0], cov, 1).T[:, 0]

    def _fit(self, x_obs, y_obs):
        """
        Description
        ----------
        Fit the model to data. This function takes in training data
        (x: features, y: targets) and fits the K matrix to that data. The
        predict function can then be used to make predictions.

        Parameters
        ----------
        x_obs: array_like
            An array containing the model features.
        y_obs: array_like
            An array containing the model targets (currently only supports
            single outputs).

        Returns
        ----------
        K: array_like
            The matrix representing the kernel distances of all train
            samples.
        """
        self.x_obs = exactly_2d(x_obs)
        self.y_obs = exactly_2d(y_obs).copy()

        if self.prior_mean is not None:
            self.y_obs -= exactly_2d(exactly_1d(self.prior_mean(self.x_obs)))

        if isinstance(self.var_l, np.ndarray):
            assert self.var_l.shape[0] == self.y_obs.shape[0], \
                "The length of the likelihood variance array does not match the number of training observations."

        K = self.kernel(x_obs, x_obs)

        identity = np.zeros(K.shape)
        idx = np.diag_indices(identity.shape[0])
        identity[idx] = self.var_l
        K += identity

        return K

    def posterior_sample(self, x_test):
        """
        Description
        ----------
        Draw a function from the fitted posterior.

        Parameters
        ----------
        x_test: array_like
            Feature input for points to draw samples for.

        Returns
        ----------
        Sample: array_like
            The values of a function sampled from the gaussian process posterior.
        """
        assert self.fitted, "Please fit the model before trying to make posterior predictions!"

        mean, cov = self.posterior_predict(x_test, True)
        return self.random.multivariate_normal(mean[:, 0], cov, 1).T[:, 0]

# ---------------------------------------------------------------------------------------------------------------------
# Gaussian Process Regression
# SOGP with Inversion
# ---------------------------------------------------------------------------------------------------------------------


class GaussianProcessInversion(GaussianProcessBase):
    """Model object for single output gaussian process (SOGP) regression."""

    def __init__(self, inv_method="inv", *args, **kwargs):
        """
        Description
        ----------
        Model object for single output gaussian process (SOGP) regression. See
        base class init docstring for documentation on additional parameters.

        Parameters
        ----------
        inv_method: string
            A string argument choosing an inversion method for matrix K when
            fitting the gaussian process.

        Returns
        ----------
        Model object
        """
        self.inv = Invert(inv_method)
        GaussianProcessBase.__init__(self, *args, **kwargs)

    def fit(self, x_obs, y_obs):
        """
        Description
        ----------
        Fit the model to data. This function takes in training data
        (x: features, y: targets) and fits the K matrix to that data. The
        predict function can then be used to make predictions. Borrows from
        base class self._fit() function.

        Parameters
        ----------
        x_obs: array_like
            An array containing the model features.
        y_obs: array_like
            An array containing the model targets (currently only supports
            single outputs).

        Returns
        ----------
        None
        """
        self.K = self._fit(x_obs, y_obs)
        self.inv_K = self.inv(self.K)
        self.fitted = True

    def posterior_predict(self, x_test, return_cov=False):
        """
        Description
        ----------
        Make predictions based on fitted model. This function takes in a set of
        test points to make predictions on and returns the mean function of the
        gaussian process and a measure of uncertainty (either covariance or
        variance).

        Parameters
        ----------
        x_test: array_like
            Feature input for points to make predictions for.
        return_cov: boolean
            If true, will return the full covariance matrix. Otherwise it will
            return the variance.

        Returns
        ----------
        Mean: array_like
            An array with the values of the mean function of the guassian
            process posterior.
        Var: array_like
            The variance around the values of the mean function of the
            gaussian process posterior. If return_cov is True then returns the
            full covariance matrix of the gaussian process posterior instead.
        """
        assert self.fitted and (self.inv_K is not None), \
            "Please fit the model before trying to make posterior predictions!"

        x_test = exactly_2d(x_test)

        # Gaussian Processes for Machine Learning Eq 2.18/2.19
        K_s = self.kernel(x_test, self.x_obs)
        K_ss = self.kernel(x_test, x_test)

        mean = K_s.dot(self.inv_K).dot(self.y_obs)
        cov = K_ss - np.dot(np.dot(K_s, self.inv_K), K_s.T)

        if self.prior_mean is not None:
            mean += self.prior_mean(x_test)
        mean = exactly_2d(mean)

        check_valid_cov(cov, self.safe)
        if return_cov:
            return mean, cov
        var = exactly_2d(np.diag(cov))
        return mean, var

# ---------------------------------------------------------------------------------------------------------------------
# Gaussian Process Regression
# SOGP Stable Cholesky
# ---------------------------------------------------------------------------------------------------------------------


class GaussianProcessCholesky(GaussianProcessBase):
    """Model object for single output gaussian process (SOGP) regression formulated for stability.."""

    def __init__(self, *args, **kwargs):
        """
        Description
        ----------
        Model object for single output gaussian process (SOGP) regression. See
        base class init docstring for documentation on additional parameters.
        Uses algorithm 2.1 (pg.19) from Gaussian Processes for Machine Learning
        for increased numerical stability and faster performance.

        Parameters
        ----------

        Returns
        ----------
        Model object
        """
        GaussianProcessBase.__init__(self, *args, **kwargs)

    def fit(self, x_obs, y_obs):
        """
        Description
        ----------
        Fit the model to data. This function takes in training data
        (x: features, y: targets) and fits the K matrix to that data. The
        predict function can then be used to make predictions. Borrows from
        base class self._fit() function.

        Parameters
        ----------
        x_obs: array_like
            An array containing the model features.
        y_obs: array_like
            An array containing the model targets (currently only supports
            single outputs).

        Returns
        ----------
        None
        """
        K = self._fit(x_obs, y_obs)

        # More numerically stable
        # Gaussian Processes for Machine Learning Alg 2.1
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.transpose(), np.linalg.solve(self.L, self.y_obs))
        self.fitted = True

    def posterior_predict(self, x_test, return_cov=False):
        """
        Description
        ----------
        Make predictions based on fitted model. This function takes in a set of
        test points to make predictions on and returns the mean function of the
        gaussian process and a measure of uncertainty (either covariance or
        variance).

        Parameters
        ----------
        x_test: array_like
            Feature input for points to make predictions for.
        return_cov: boolean
            If true, will return the full covariance matrix. Otherwise it will
            return the variance.

        Returns
        ----------
        Mean: array_like
            An array with the values of the mean function of the guassian
            process posterior.
        Var: array_like
            The variance around the values of the mean function of the
            gaussian process posterior. If return_cov is True then returns the
            full covariance matrix of the gaussian process posterior instead.
        """
        assert self.fitted and (self.alpha is not None) and (self.L is not None), \
            "Please fit the model before trying to make posterior predictions!"

        x_test = exactly_2d(x_test)

        # Gaussian Processes for Machine Learning Eq 2.18/2.19
        K_ = self.kernel(self.x_obs, x_test)
        K_ss = self.kernel(x_test, x_test)
        V = np.linalg.solve(self.L, K_)

        mean = np.dot(K_.transpose(), self.alpha)
        cov = K_ss - np.dot(V.transpose(), V)

        if self.prior_mean is not None:
            mean += self.prior_mean(x_test)
        mean = exactly_2d(mean)

        check_valid_cov(cov, self.safe)
        if return_cov:
            return mean, cov
        var = exactly_2d(np.diag(cov))
        return mean, var
