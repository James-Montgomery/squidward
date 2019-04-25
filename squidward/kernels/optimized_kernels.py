"""
This module contains 'optimized' kernels that will run much quicker
than the more general kernel format of Kernel_Base + Distance_Function.
"""

import numpy as np

from squidward.utils import exactly_2d

# TODO: Better docstrings

# ---------------------------------------------------------------------------------------------------------------------
# Radial Basis Function
# ---------------------------------------------------------------------------------------------------------------------


class RBF_Kernel(object):
    """Radial Basis Function Kernel"""

    def __init__(self, lengthscale, var_k):
        """
        Description
        ----------
        Radial basis function (rbf) kernel.

        Parameters
        ----------
        lengthscale: Float
            The lengthscale of the rbf function that detrmins the radius around
            which the value of an observation imapcts other observations.
        var_k: Float
            The kernel variance or amplitude. This can be thought of as the maximum
            value that the rbf function can take.

        Returns
        ----------
        kernel object
        """
        self.lengthscale = lengthscale
        self.var_k = var_k
        if lengthscale <= 0.0:
            raise Exception("Lengthscale parameter must be greater than zero.")
        if var_k <= 0.0:
            raise Exception("Kernel variance parameter must be greater than zero.")

    def __call__(self, alpha, beta):
        """
        Description
        ----------
        Calls the kernel object.

        Parameters
        ----------
        alpha: array_like
            The first vector to compare.
        beta: array_like
            The second vector to compare.

        Returns
        ----------
        A array representing the covariance between points in
        the vectors alpha and beta.
        """
        alpha, beta = exactly_2d(alpha), exactly_2d(beta)
        if alpha.shape[1] != beta.shape[1]:
            raise Exception("Input arrays have differing number of features.")
        distance = np.sum(alpha**2, 1).reshape(-1, 1) + np.sum(beta**2, 1) - 2*np.dot(alpha, beta.T)
        gamma = -0.5/self.lengthscale**2
        return self.var_k*np.exp(gamma * distance)

# ---------------------------------------------------------------------------------------------------------------------
# A different kernel
# ---------------------------------------------------------------------------------------------------------------------
