"""
This script contains code for basic gaussian process classification. A
classification model can be created by calling one of these classes to create
a model object.
"""

import numpy as np
from squidward import gpr
from squidward.utils import exactly_2d, softmax, reversehot

np.seterr(over="raise")

# ---------------------------------------------------------------------------------------------------------------------
# Gaussian Process Classification
# ---------------------------------------------------------------------------------------------------------------------

class GaussianProcess(object):
    """Model object for single output gaussian process classification."""

    def __init__(self, n_classes=None, kernel=None, var_l=1e-15, show_warnings=True):
        """
        Description
        ----------
        Model object for one vs all implementation of gaussian process classification.

        Parameters
        ----------
        n_classes: int
            The total number of expected classes.
        kernel : kernel object
            An object with an associated function k that takes in 2 arrays and
            returns a valid K matrix. Valid K matricies are positive
            semi-definite and not singular.
        var_l: float
            The likelihood variance of the process. Currently only supports
            scalars for homoskedastic regression.
        inv_method: string
            A string argument choosing an inversion method for matrix K when
            fitting the gaussian process.
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

        self.x_obs = None
        self.y_obs = None

        self._fitted = False

        self.n_classes = n_classes
        self._predictors = []

        assert self.n_classes is not None, \
            "Please specify the number of classes."
        assert isinstance(self.n_classes, int), \
            "n_classes must be an integer."
        assert self.n_classes > 1, \
            "n_classes must be greater than one."
        assert self.kernel is not None, \
            "Model object must be instantiated with a valid kernel object."
        if isinstance(self.var_l, (list, np.ndarray)):
            raise Exception("GP classification does not support heteroscedastic noise.")
        assert self.var_l >= 0.0, \
            "Invalid likelihood variance argument."

    def prior_predict(self, x_test, logits=False):
        """
        While each regressor in the one vs. all gaussian process classifier has
        a prior. The softmax over their collective priors has no actual
        interpretation and is not supported by this package. You can, however,
        sample from their collective priors.
        """
        raise NotImplementedError("Priors not supported for one vs. all gaussian process classification.")

    def prior_sample(self, x_test, logits=False):
        """
        Description
        ----------
        Make predictions based on samples from the prior of the un_fitted
        model. This function takes in a set of test points to make predictions
        on and returns the mean function of the gaussian process and a measure
        of uncertainty (either covariance or variance).

        Parameters
        ----------
        x_test: array_like
            Feature input for points to make predictions for.
        logits: boolean
            If True, will return the means and variances of the one vs. all
            gaussian processes for each class. If False, returns the softmax
            class probabilities of the classes.

        Returns
        ----------
        Softmax Prob: array_like
            The softmax probabilities of each class for every test sample.
        Means: array_like
            The means of each one vs. all gaussian process for each class.
        Var: array_like
            The variance around the mean of each one vs. all gaussian process
        """

        x_test = exactly_2d(x_test)
        samples = []

        if self.n_classes == 2:
            _range = range(1)
        else:
            _range = range(self.n_classes)

        for _ in _range:
            model = gpr.GaussianProcessCholesky(kernel=self.kernel, var_l=self.var_l, \
                                                show_warnings=self.safe)
            sample = model.prior_sample(x_test)
            samples.append(sample)

        samples = np.array(samples).T

        if logits:
            return samples

        samples = softmax(samples)
        return exactly_2d(samples)

    def fit(self, x_obs, y_obs):
        """
        Description
        ----------
        Fit the model to data. This function takes in training data
        (x_obs: features, y_obs: targets/classes) and fits the K matrix to that data. The
        predict function can then be used to make predictions.

        Parameters
        ----------
        x_obs: array_like
            An array containing the model features.
        y_obs: array_like
            An array containing the model targets. Targets should be classes
            counting up from a zero index using integers.
            (i.e. y_obs = [0,1,2,0,2,...])

        Returns
        ----------
        None
        """
        self.x_obs = exactly_2d(x_obs)
        y_obs = reversehot(y_obs).copy()
        self.y_obs = exactly_2d(y_obs)

        if self.n_classes < np.unique(self.y_obs).shape[0]:
            raise Exception("More classes in ytrain than specified in model object.")

        if self.n_classes == 2:
            _range = range(1)
        else:
            _range = range(self.n_classes)

        for i in _range:
            y_obs_class = np.where(self.y_obs == i, 1, -1)
            model = gpr.GaussianProcessCholesky(kernel=self.kernel, var_l=self.var_l, \
                                                show_warnings=self.safe)
            model.fit(x_obs, y_obs_class.T)
            self._predictors.append(model)
        self._fitted = True

    def posterior_predict(self, x_test, logits=False):
        """
        Description
        ----------
        Make predictions based on _fitted model. This function takes in a set of
        test points to make predictions on and returns the mean function of the
        gaussian process and a measure of uncertainty (either covariance or
        variance).

        Parameters
        ----------
        x_test: array_like
            Feature input for points to make predictions for.
        logits: boolean
            If True, will return the means and variances of the one vs. all
            gaussian processes for each class. If False, returns the softmax
            class probabilities of the classes.

        Returns
        ----------
        Softmax Prob: array_like
            The softmax probabilities of each class for every test sample.
        Means: array_like
            The means of each one vs. all gaussian process for each class.
        Var: array_like
            The variance around the mean of each one vs. all gaussian process
        """
        assert self._fitted and (len(self._predictors) > 0), \
            "Please fit the model before trying to make posterior predictions!"

        x_test = exactly_2d(x_test)
        means = []
        variances = []
        for model in self._predictors:
            mean, var = model.posterior_predict(x_test)
            means.append(mean)
            variances.append(var)

        if logits:
            means = np.array(means)[:, :, 0].T
            variances = np.array(variances)[:, :, 0].T
            return exactly_2d(means), exactly_2d(variances)

        means = softmax(np.asarray(means)[:, :, 0].T)
        return exactly_2d(means)

    def posterior_sample(self, x_test, logits=False):
        """
        Description
        ----------
        Make predictions based on samples from the posterior of the _fitted
        model. This function takes in a set of test points to make predictions
        on and returns the mean function of the gaussian process and a measure
        of uncertainty (either covariance or variance).

        Parameters
        ----------
        x_test: array_like
            Feature input for points to make predictions for.
        logits: boolean
            If True, will return the means and variances of the one vs. all
            gaussian processes for each class. If False, returns the softmax
            class probabilities of the classes.

        Returns
        ----------
        Softmax Prob: array_like
            The softmax probabilities of each class for every test sample.
        Means: array_like
            The means of each one vs. all gaussian process for each class.
        Var: array_like
            The variance around the mean of each one vs. all gaussian process
        """
        assert self._fitted and (len(self._predictors) > 0), \
            "Please fit the model before trying to make posterior predictions!"

        x_test = exactly_2d(x_test)
        samples = []
        for model in self._predictors:
            sample = model.prior_sample(x_test)
            samples.append(sample)

        samples = np.array(samples).T

        if logits:
            return samples

        samples = softmax(samples)
        return exactly_2d(samples)
