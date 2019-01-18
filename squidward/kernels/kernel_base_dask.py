"""
Contains code for the base kernel object used when making kernels for
gaussian process modeling with multiprocessing functionality using Dask.
"""

import numpy as np
from distributed import Client

def worker(i, alpha_element, beta, m_len, distance_function):
    """
    Worker function for kernel_base_multiprocessing.
    """
    output = np.full(m_len, 0.0)
    for j in range(m_len):
        output[j] = distance_function(alpha_element, beta[j])
    return output.reshape(-1)

class Kernel(object):
    """
    This class is the base class for a kernel object. It basically takes the
    input distance fucntion and finds the the distance between all vectors in
    two lists and returns that matrix as a covariance matrix.
    """
    def __init__(self, distance_function, client=None):
        """
        Description
        ----------
        Kernel base class for creating GP kernels.

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
        Crude multiprocessing implementation of kernel evaluation
        Memory hungry
        """

        n_len, m_len = alpha.shape[0], beta.shape[0]

        cov = np.full((n_len, m_len), None)

        [distributed_beta] = self.client.scatter([beta], broadcast=True)
        futures = [self.client.submit(worker, i, alpha[i], distributed_beta, m_len, self.distance_function) for i in range(n_len)]
        results = self.client.gather(futures)
        for i, row in enumerate(results):
            cov[i,:] = row

        return cov
