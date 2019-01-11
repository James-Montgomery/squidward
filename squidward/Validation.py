import numpy as np
import scipy.stats as st
from squidward.Utils import atmost_1d, check_valid_cov, is_invertible

np.seterr(over="raise")

def preprocess(func):
    def wrapper(*args, **kwargs):
        if args:
            p, y = args[0], args[1]
            p, y = atmost_1d(p), atmost_1d(y)
        if kwargs:
            p, y  = kwargs['p'], kwargs['y']
            p, y = atmost_1d(p), atmost_1d(y)
        if y.shape[0] != p.shape[0]:
            raise Exception("Number of predictions does not match number of targets")
        return func(p=p,y=y)
    return wrapper

def likelihood(mean, cov, y, log=False, allow_singular=False):
    """
    """
    mean = atmost_1d(mean)
    check_valid_cov(cov)
    if not is_invertible(cov):
        warnings.warn('Matrix has high condition. Inverting matrix may result in errors.')
    if log == False:
        return st.multivariate_normal(mean, cov, allow_singular=allow_singular).pdf(y)
    return st.multivariate_normal(mean, cov, allow_singular=allow_singular).logpdf(y)

@preprocess
def rmse(p, y):
    """
    """
    return np.sqrt( np.sum((p - y)**2) / y.shape[0] )

@preprocess
def acc(p, y):
    """
    """
    return y[y == p].shape[0] / y.shape[0]

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
