import numpy as np
from scipy.linalg import norm
from squidward.Utils import atleast_2d

np.seterr(over="raise")

def rbf(l, var_k, method=None):
    """
    """
    if l <= 0.0:
        raise Exception("Lengthscale parameter must be greater than zero")
    if var_k <= 0.0:
        raise Exception("Kernel variance parameter must be greater than zero")

    def _standard_dist(alpha, beta):
        alpha, beta = atleast_2d(alpha), (beta)
        d = np.sum((alpha - beta)**2)
        amp = -0.5/l**2
        return var_k * np.exp( amp*d )

    def dist(method):
        return _standard_dist

    return dist(method)
