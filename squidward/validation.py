"""
Function for basic model validation. Sklearn has very nice implementations
of a much wider variety of model validation metrics.
"""

import warnings
import numpy as np
import scipy.stats as st
from squidward.utils import atmost_1d, check_valid_cov, is_invertible

np.seterr(over="raise")

def preprocess(func):
    """
    Decorator function used for preprocessing for classification
    validation metrics.
    """
    def wrapper(*args, **kwargs):
        """
        Wrapper function for decorator.
        """
        if args:
            prediction, target = args[0], args[1]
            prediction, target = atmost_1d(prediction), atmost_1d(target)
        if kwargs:
            prediction, target = kwargs['prediction'], kwargs['target']
            prediction, target = atmost_1d(prediction), atmost_1d(target)
        if prediction.shape[0] != target.shape[0]:
            raise Exception("Number of predictions does not match number of targets.")
        return func(prediction=prediction, target=target)
    return wrapper

def likelihood(mean, cov, target, log=False, allow_singular=False):
    """
    Multivariate normal likelihood for a set of data given the
    parameters of a gaussian process.
    """
    mean = atmost_1d(mean)
    check_valid_cov(cov)
    if not is_invertible(cov):
        warnings.warn('Matrix has high condition. Inverting matrix may result in errors.')
    if not log:
        return st.multivariate_normal(mean, cov, allow_singular=allow_singular).pdf(target)
    return st.multivariate_normal(mean, cov, allow_singular=allow_singular).logpdf(target)

@preprocess
def rmse(prediction, target):
    """
    Calculate of the root mean squared error of univariate regression model.
    """
    return np.sqrt(np.sum((prediction-target) **2)/target.shape[0])

@preprocess
def acc(prediction, target):
    """
    Calculate the accuracy of univariate classification problem.
    """
    return target[target == prediction].shape[0]/target.shape[0]

# TODO: add the following methods
# brier_score
# precision
# recall
# roc_auc
#posterior_checkes
