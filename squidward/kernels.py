"""
This script contains code to build basic kernel objects for gaussian process
model. The kernel object is passed to the model object. Feel free to write
custom kernel rather than using the kernel provided here. Use the format of
these kernel as a template to write your custom kernel.
"""

import numpy as np

np.seterr(over='raise')

# future optimization
# https://stackoverflow.com/questions/47271662/what-is-the-fastest-way-to-compute-an-rbf-kernel-in-python

class RBF(object):
    """
    Kernel object for radial basis function/squared exponential kernel.
    """
    def __init__(self, var_k, l):
        """
        Description
        ----------
        Kernel object for radial basis function/squared exponential kernel.

        Parameters
        ----------
        l: float
            lengthscale parameter for rbf kernel
        var_k: float
            kernel variance perameter

        Returns
        ----------
        Kernel object
        """
        self.var_k = var_k
        self.l = l

    def _k(self, alpha, beta):
        """
        The meat and potatoes of the rbf kernel math.

                                n
        k(a, b) = exp( -.5/l * sum (a-b)^2 )
                              i = 1

        """
        alpha = alpha.reshape(-1, 1)
        beta = beta.reshape(-1, 1)
        zed = np.sum(alpha**2, 1).reshape(-1, 1) + np.sum(beta**2, 1) -2.0*alpha.dot(beta.T)
        alpha = (1.0/(self.l**2))
        return self.var_k * np.exp(-0.5*alpha*zed)

    def k(self, alpha, beta):
        """
        Description
        ----------
        This function returns the kernel output. If there are multiple dimensions
        to compare, returns a multiplicative kernel over the dimensions.

        Parameters
        ----------
        alpha: array_like
            first array to compare
        beta: array_like
            second array to compare

        Returns
        ----------
        k: array_like
            the pair-wise distances from the kernel output
        """
        # if one dimensional return k(alpha, beta)
        if len(alpha.shape) == 1:
            return self._k(alpha, beta)
        # ele return the product of a kernel for every dimension
        output = 1
        for i in range(alpha.shape[1]):
            output *= self._k(alpha[:, i], beta[:, i])
        return output
