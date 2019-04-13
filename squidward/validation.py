"""
Function for basic model validation. Sklearn has very nice implementations
of a much wider variety of model validation metrics.
"""

import numpy as np
from squidward.utils import exactly_1d

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
            predictions, targets = args[0], args[1]
            predictions, targets = exactly_1d(predictions), exactly_1d(targets)

        if kwargs:
            predictions, targets = kwargs['predictions'], kwargs['targets']
            predictions, targets = exactly_1d(predictions), exactly_1d(targets)

        if predictions.shape[0] != targets.shape[0]:
            raise Exception("Number of predictions does not match number of targets.")

        return func(predictions=predictions, targets=targets)

    return wrapper

@preprocess
def rmse(predictions, targets):
    """
    Description
    ----------
    Calculate of the root mean squared error of univariate regression model.

    Parameters
    ----------
    predictions: array-like
        description
    targets: array-like
        description

    Returns
    ----------
    """
    return np.sqrt(np.sum((predictions-targets) **2)/targets.shape[0])

@preprocess
def acc(predictions, targets):
    """
    Description
    ----------
    Calculate the accuracy of univariate classification problem.

    Parameters
    ----------

    Returns
    ----------
    """
    return 1.0 * targets[targets == predictions].shape[0] / targets.shape[0]

# TODO: add the following methods
# brier_score
# precision
# recall
# roc_auc
# posterior_checks
