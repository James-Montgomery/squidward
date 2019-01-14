"""
Contains code for the base kernel object used when making kernels for
gaussian process modeling.
"""

import multiprocessing
import numpy as np

np.seterr(over="raise")

class Kernel(object):
    """
    This class is the base class for a kernel object. It basically takes the
    input distance fucntion and finds the the distance between all vectors in
    two lists and returns that matrix as a covariance matrix.
    """
    def __init__(self, distance_function, method='k1', pool_size=1):
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
        pool_size: Int
            The number of workers if the method selected supports
            multi-processing.

        Returns
        ----------
        Model object
        """
        self.distance_function = distance_function
        self.pool = multiprocessing.Pool(processes=pool_size)
        if method == 'k1':
            self.k = self._k1
        # elif method == 'k2':
        #     self.k = self._k2
        # elif method == 'k3':
        #     self.k = self._k3
        else:
            raise Exception("Invalid argument for kernel method")

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
