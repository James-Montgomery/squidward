"""
Contains code for the base kernel object used when making kernels for
gaussian process modeling.
"""

import numpy as np

np.seterr(over="raise")

def array_equal(alpha, beta):
    """
    Function returns true if two arrays are identical.
    """
    return alpha.shape == beta.shape and np.all(np.sort(alpha) == np.sort(beta))

class Kernel(object):
    """
    This class is the base class for a kernel object. It basically takes the
    input distance fucntion and finds the the distance between all vectors in
    two lists and returns that matrix as a covariance matrix.
    """
    def __init__(self, distance_function, method='k1'):
        """
        Description
        ----------
        Kernel base class for creating GP kernels.

        Parameters
        ----------
        distance_function : Function
            A function that takes in two vectors and returns a float
            representing the distance between them.
        method: String
            The method used for iterating over the input vectors to arrive
            at the covariance matrix.

        Returns
        ----------
        Model object
        """
        self.distance_function = distance_function
        if method == 'k1':
            self._k = self._k1
        elif method == 'k2':
            self._k = self._k2
        else:
            raise Exception("Invalid argument for kernel method")

    def __call__(self, alpha, beta):
        """
        Parameters
        ----------
        alpha: array-like
            The first array to compare.
        beta: array-like
            The second array to compare.
        """
        return self._k(alpha, beta)

    def _k1(self, alpha, beta):
        """
        Implementation inspired by scipy.spacial.distance cdist v1.2.0
        For loop through every index i,j for input vectors alpha_i and beta_j
        """
        # lengths of each vector to compare
        n_len, m_len = alpha.shape[0], beta.shape[0]
        # create an empty array to fill with element wise vector distances
        cov = np.full((n_len, m_len), 0.0)
        # loop through each vector
        for i in range(n_len):
            for j in range(m_len):
                # assign distances to each element in covariance matrix
                cov[i, j] = self.distance_function(alpha[i], beta[j])
        return cov

    def _k2(self, alpha, beta):
        """
        Implementation that exploits covariance symmetry when possible. Good
        for fitting and testing on larger datasets.
        """
        # lengths of each vector to compare
        n_len, m_len = alpha.shape[0], beta.shape[0]
        # if comparing an array against itself exploit symmetry
        if array_equal(alpha, beta):
            # create an empty array to fill with element wise vector distances
            cov = np.full((n_len, m_len), 0.0)
            # loop through each vector
            for i in range(n_len):
                for j in range(i, m_len):
                    # assign distances to each element in covariance matrix
                    cov[i, j] = cov[j, i] = self.distance_function(alpha[i], beta[j])
            return cov
        return self._k1(alpha, beta)
