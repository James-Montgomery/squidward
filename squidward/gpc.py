"""
This script contains code for basic gaussian process classification. A
classification model can be created by calling one of these classes to create
a model object.
"""

import numpy as np
import squidward.gpr as gpr
from squidward.utils import atleast_2d, sigmoid, softmax

np.seterr(over="raise")

class GaussianProcess(object):
    """
    Model object for single output gaussian process classification.
    """
    def __init__(self, kernel=None, var_l=1e-15, inv_method="inv"):
        """
        Description
        ----------
        Model object for one vs all implementation of gaussian process classification.

        Parameters
        ----------
        kernel : kernel object
            An object with an associated function k that takes in 2 arrays and
            returns a valid K matrix. Valid K matricies are positive
            semi-definite and not singular.
        var_l: float
            The liklihood variance of the process. Currently only supports
            scalars for homoskedastic regression.
        inv_method: string
            A string argument choosing an inversion method for matrix K when
            fitting the gaussian process.

        Returns
        ----------
        Model object
        """
        self.var_l = var_l
        self.x_obs = None
        self.y_obs = None
        self.kernel = kernel
        self.inv_method = inv_method
        self.predictors = []
        self.n_classes = None
        self.fitted = False
        assert self.kernel is not None, \
            "Model object must be instantiated with a valid kernel object."
        assert self.var_l >= 0.0, \
            "Invalid likelihood variance argument."

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
            counting up from a zero idnex using integers.
            (i.e. y_obs = [0,1,2,0,2,...])

        Returns
        ----------
        None
        """
        self.x_obs = atleast_2d(x_obs)
        if len(y_obs.shape) > 1:
            if y_obs.shape[1] > 1:
                y_obs = y_obs.argmax(axis=1)
        self.y_obs = atleast_2d(y_obs)
        self.n_classes = np.unique(self.y_obs).shape[0]
        for i in range(self.n_classes):
            y_obs_class = np.where(self.y_obs == i, 1, -1)
            model = gpr.GaussianProcess(kernel=self.kernel, var_l=self.var_l, \
                                        inv_method=self.inv_method)
            model.fit(x_obs, y_obs_class)
            self.predictors.append(model)
        self.fitted = True
        return None

    def posterior_predict(self, x_test, logits=False):
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
        assert self.fitted, "Please fit the model before trying to make posterior predictions!"

        x_test = atleast_2d(x_test)
        means = []
        variances = []
        for model in self.predictors:
            mean, var = model.posterior_predict(x_test)
            means.append(mean)
            variances.append(var)
        if logits:
            means = np.array(means).T[0]
            variances = np.array(variances).T[0]
            return means, variances
        means = np.array(means).T[0]
        means = softmax(sigmoid(means))
        return atleast_2d(means.argmax(axis=1))

    def prior_predict(self, x_test, logits=False):
        """
        While each regressor in the one vs. all gaussian process classifier has
        a prior. The softmax over their collective prior has no actual
        interpretarion and is not supported by this package.
        """
        raise NotImplementedError("Priors not supported for one vs. all gaussian process classification.")

    def posterior_sample(self, x_test, logits=False):
        """
        Description
        ----------
        Make predictions based on samples from the posterior of the fitted
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
        assert self.fitted, "Please fit the model before trying to make posterior predictions!"

        x_test = atleast_2d(x_test)
        means = []
        variances = []
        for model in self.predictors:
            mean, var = model.posterior_sample(x_test)
            means.append(mean)
            variances.append(var)
        if logits:
            means = np.array(means).T[0]
            variances = np.array(variances).T[0]
            return means, variances
        means = np.array(means).T[0]
        means = softmax(sigmoid(means))
        return atleast_2d(means.argmax(axis=1))

    def prior_sample(self, x_test, logits=False):
        """
        Description
        ----------
        Make predictions based on samples from the prior of the unfitted
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
        assert self.fitted, "Please fit the model before trying to make posterior predictions!"

        x_test = atleast_2d(x_test)
        means = []
        variances = []
        for model in self.predictors:
            mean, var = model.prior_sample(x_test)
            means.append(mean)
            variances.append(var)
        if logits:
            means = np.array(means).T[0]
            variances = np.array(variances).T[0]
            return means, variances
        means = np.array(means).T[0]
        means = softmax(sigmoid(means))
        return atleast_2d(means.argmax(axis=1))
