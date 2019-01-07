import numpy as np
import scipy.stats as st
from squidward.Utils import atmost_1d, check_valid_cov

def rmse(p, y):
    """
    """
    p = atmost_1d(p)
    y = atmost_1d(y)
    return np.sqrt(np.sum((p - y)**2)/y.shape[0])

def likelihood(mean, cov, y, log=False, allow_singular=False):
    """
    """
    mean = atmost_1d(mean)
    check_valid_cov(cov)
    if log == False:
        return st.multivariate_normal(mean, cov, allow_singular=allow_singular).pdf(y)
    else:
        return st.multivariate_normal(mean, cov, allow_singular=allow_singular).logpdf(y)

def acc(p, y):
    """
    """
    if y.shape[0] <= 1:
        return y.T[y.T == p].shape[0] / y.shape[0]
    else:
        return y[y == p].shape[0] / y.shape[1]

def brier_score():
    """
    """
    raise NotImplementedError()

def precision():
    """
    """
    raise NotImplementedError()

def recall():
    """
    """
    raise NotImplementedError()

def roc_auc():
    """
    """
    raise NotImplementedError()

def posterior_checks():
    """
    """
    raise NotImplementedError()

def variational_inference():
    """
    """
    raise NotImplementedError()
