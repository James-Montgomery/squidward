"""
Contains code for the base kernel object used when making kernels for
gaussian process modeling with multiprocessing functionality.
"""

import multiprocessing
import numpy as np
from squidward.utils import worker, exactly_2d

np.seterr(over="raise")

# watch out for error:
# too many open files
# meaning that you have spun up too many processes
# TODO: Let user control the number of processes
pool_size = multiprocessing.cpu_count()
pool = multiprocessing.Pool(processes=pool_size)

class Kernel(object):
    """Base class for Kernel object."""

    def __init__(self, distance_function):
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
        self.distance_function = distance_function

    def __call__(self, alpha, beta):
        """
        Crude multiprocessing implementation of kernel evaluation.
        """
        alpha, beta = exactly_2d(alpha), exactly_2d(beta)
        if alpha.shape[1] != beta.shape[1]:
            raise Exception("Input arrays have differing number of features.")
        # lengths of each vector to compare
        n_len, m_len = alpha.shape[0], beta.shape[0]
        # create an empty array to fill with element wise vector distances
        cov = np.full((n_len, m_len), 0.0)

        def log_result(result):
            """
            Assigns worker outputs to their spots in the covariance matrix.
            """
            i, row = result
            cov[i, :] = row

        # loop through each vector and keep track of result flags
        result_flags = np.full((n_len), None)
        for i in range(n_len):
            # arguments for the worker
            args = (i, alpha[i], beta, m_len, self.distance_function)
            # put task in queue for worker pool
            result_flag = pool.apply_async(worker, args=args, callback=log_result)
            result_flags[i] = result_flag

        # wait for processes to finish
        for result_flag in result_flags:
            result_flag.wait()

        # for debugging purposes
        #result_list = []
        #for i in range(n_len):
        #    args = (i, alpha[i], beta, m_len, self.distance_function)
        #    result_list.append( pool.apply_async(worker, args = args) )

        return cov
