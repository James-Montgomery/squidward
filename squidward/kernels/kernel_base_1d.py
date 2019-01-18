
import numpy as np
from squidward.utils import atmost_1d, exactly_2d

class RBF_Kernel(object):
    def __init__(self, lengthscale, var_k):
        self.lengthscale = lengthscale
        self.var_k = var_k
        if lengthscale <= 0.0:
            raise Exception("Lengthscale parameter must be greater than zero.")
        if var_k <= 0.0:
            raise Exception("Kernel variance parameter must be greater than zero.")
    def __call__(self, alpha, beta):
        alpha, beta = atmost_1d(alpha), atmost_1d(beta)
        alpha, beta = exactly_2d(alpha), exactly_2d(beta)
        distance = np.sum(alpha**2,1).reshape(-1,1) + np.sum(beta**2,1) - 2*np.dot(alpha, beta.T)
        gamma = -0.5/self.lengthscale**2
        return self.var_k * np.exp(gamma * distance)
