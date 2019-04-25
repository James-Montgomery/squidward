"""
Distance functions to define how "far" apart two vectors are.
"""

import numpy as np
from squidward.utils import exactly_1d

np.seterr(over="raise")

# ---------------------------------------------------------------------------------------------------------------------
# Radial Basis Function
# ---------------------------------------------------------------------------------------------------------------------


class RBF(object):
    """Class for radial basis fucntion distance measure."""

    def __init__(self, lengthscale, var_k):
        """
        Description
        ----------
        Radial basis function (rbf) distance measure between vectors/arrays.

        kSE(x,x′)=σ2exp(−(x−x′)/2ℓ^2)

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
        distance object
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
        alpha, beta = exactly_1d(alpha), exactly_1d(beta)
        distance = np.sum((alpha - beta)**2)
        amp = -0.5/self.lengthscale**2
        return self.var_k*np.exp(amp*distance)

# ---------------------------------------------------------------------------------------------------------------------
# Linear Kernel
# --------------------------------------------------------------------------------------------------------------------


class Linear(object):
    """Class for radial basis fucntion distance measure."""

    def __init__(self, c, var_b, var_k):
        """
        Description
        ----------
        Linear distance measure between vectors/arrays.

        kLin(x,x′)=σ2b+σ2v(x−c)(x′−c)

        Parameters
        ----------
        c: Float
            The kernel offset.
        var_b: Float
            The constant variance.
        var_k: Float
            The kernel variance.

        Returns
        ----------
        distance object
        """
        self.c = c
        self.var_b = var_b
        self.var_k = var_k

        assert self.var_b > 0.0, "Invalid argument"
        assert self.var_k > 0.0, "Invalid argument"

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
        alpha, beta = exactly_1d(alpha), exactly_1d(beta)
        return self.var_b + self.var_k*(alpha - self.c)*(beta - self.c)