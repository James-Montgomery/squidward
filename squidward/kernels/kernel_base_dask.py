"""
Contains code for the base kernel object used when making kernels for
gaussian process modeling with multiprocessing functionality using Dask.
"""

import numpy as np
from distributed import Client
from squidward.utils import exactly_2d

def worker(alpha_element, beta, m_len, distance_function):
    """
    Worker function for kernel_base_dask.
    """
    output = np.full(m_len, 0.0)
    for j in range(m_len):
        output[j] = distance_function(alpha_element, beta[j])
    return output.reshape(-1)

class Kernel(object):
    """Base class for Kernel object."""

    def __init__(self, distance_function, client=None):
        """
        Description
        ----------
        This class is the base class for a kernel object. It basically takes the
        input distance function and finds the the distance between all vectors in
        two lists and returns that matrix as a covariance matrix.

        Parameters
        ----------
        distance_function : Function
            A function that takes in two vectors and returns a float
            representing the distance between them.

        Returns
        ----------
        Model object
        """
        if client is None:
            # processes = False to prevent
            # too many files open error on import
            self.client = Client(processes=False)
        else:
            self.client = client
        self.distance_function = distance_function

    def __call__(self, alpha, beta):
        """
        Multiprocessing implementation of kernel evaluation using Dask to
        handle the pool of workers.
        """
        alpha, beta = exactly_2d(alpha), exactly_2d(beta)
        if alpha.shape[1] != beta.shape[1]:
            raise Exception("Input arrays have differing number of features.")
        # lengths of each vector to compare
        n_len, m_len = alpha.shape[0], beta.shape[0]
        # create an empty array to fill with element wise vector distances
        cov = np.full((n_len, m_len), None)
        # loop through each vector and put future object into list
        scattered_beta = self.client.scatter(beta)
        futures = [self.client.submit(worker, alpha[i], scattered_beta, m_len, self.distance_function) for i in range(n_len)]
        # get futures from futures list
        results = self.client.gather(futures)
        # assign futures to covairance matrix
        for i, row in enumerate(results):
            cov[i, :] = row

        return cov
