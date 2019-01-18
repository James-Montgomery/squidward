"""
Distance functions to define how "far" apart two vectors are.
"""

import numpy as np
from squidward.utils import atleast_2d

np.seterr(over="raise")

class RBF(object):
    """Class for radial basis fucntion distance measure."""
    def __init__(self, lengthscale, var_k, method=None):
        """
        Description
        ----------
        Radial basis function (rbf) distance measure between vectors/arrays.

        Parameters
        ----------
        lengthscale: Float
            The lengthscale of the rbf function that detrmins the radius around
            which the value of an observation imapcts other observations.
        var_k: Float
            The kernel variance or amplitude. This can be thought of as the maximum
            value that the rbf function can take.
        method: String
            The particular implementation method for rbf calculation.

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
        if method is None:
            self.distance = self._standard_dist
        else:
            raise Exception("Invalid method.")

    def __call__(self, alpha, beta):
        """
        Calls the distance fucntion defined by the initializing method.
        """
        return self.distance(alpha, beta)

    def _standard_dist(self, alpha, beta):
        """
        Description
        ----------
        Homebrew implementation of rbf.

        k(x_i, x_j) = var_k * exp( -.5/l^2 * sum( sqrt( (a-b)^2 ) ) )
                    = var_k * exp( g * |a-b| )

        Returns
        ----------
        distance: Float
        """
        alpha, beta = atleast_2d(alpha), (beta)
        #distance = np.linalg.norm(alpha-beta)
        distance = np.sqrt(np.sum((alpha - beta)**2))
        # same as np.sum((alpha - beta)**2)
        amp = -0.5/self.lengthscale**2
        return self.var_k*np.exp(amp*distance)
