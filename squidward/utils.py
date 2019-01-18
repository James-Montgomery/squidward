"""
This script contains code for useful data transformations and data checks
used in the other modules of squidward.
"""

# run np.show_config() to see
# if numpy is runniing with MKL
# backend

import sys
import warnings
import functools
import numpy as np
import scipy.linalg as la
from scipy.special import expit
import multiprocessing

# watch out for error:
# too many open files
# meaning that you have spun up too many processes
# only need 3 processes for LU decomp inversion
pool = multiprocessing.Pool(processes=3)

np.seterr(over="raise")

def sigmoid(z, ingore_overflow=False):
    """
    Function to return the sigmoid transformation for every
    term in a vector.
    """
    try:
        return 1.0 / (1.0 + np.exp(-z))
    except Exception as e:
        if "overflow encountered in exp" in str(e):
            if ingore_overflow:
                return 1.0 / (1.0 + expit(-z))
        raise e

def softmax(z):
    """
    Function to return the softmax transformation over an
    input vector.
    """
    return z / z.sum(axis=1).reshape(-1, 1)

def is_invertible(arr, strength='condition'):
    """
    Function to return True is matrix is safely invertible and
    False is the matrix is not safely invertable.
    """
    if strength == 'cramer':
        return np.linalg.det(arr) == 0.0
    if strength == 'rank':
        return arr.shape[0] == arr.shape[1] and np.linalg.matrix_rank(arr) == arr.shape[0]
    return 1.0 / np.linalg.cond(arr) >= sys.float_info.epsilon

def check_valid_cov(cov):
    """
    Function to do safety checks on covariance matrices.
    """
    if not is_invertible(cov):
        warnings.warn('Cov has high condition. Inverting matrix may result in errors.')
    var = np.diag(cov)
    if var[var < 0].shape[0] != 0:
        raise Exception('Negative values in diagonal of covariance matrix.\nLikely cause is kernel inversion instability.\nCheck kernel variance.')

def exactly_2d(arr):
    """
    Function to ensure that an array has a least 2 dimensions. Used to
    formalize output / input dimensions for certain functions.
    """
    if len(arr.shape) == 1:
        return arr.reshape(-1, 1)
    if len(arr.shape) == 2:
        if arr.shape[0] == 1:
            return arr.reshape(-1, 1)
        else:
            return arr
    if len(arr.shape) == 3:
        if arr.shape[0] == 1:
            return arr[0,:,:]
        if arr.shape[2] == 1:
            return arr[:,:,0]
        raise Exception("Invalid array shape.")
    if len(arr.shape) > 3:
        raise Exception("Invalid array shape.")
    raise Exception("Invalid array shape.")

def atmost_1d(arr):
    """
    Function to ensure that an array has a most 1 dimension. Used to
    formalize output / input dimensions for certain functions.
    """
    if len(arr.shape) == 1:
        return arr
    if len(arr.shape) == 2:
        if arr.shape[0] == 1:
            return arr[0, :]
        if arr.shape[1] == 1:
            return arr[:, 0]
    raise Exception("Not appropriate input shape.")

def make_grid(coordinates=(-10, 10, 1)):
    """
    Returns a square grid of points determined by the input coordinates
    using nump mgrid. Used in visualization fucntions.
    """
    min_, max_, grain = coordinates
    if min_ >= max_:
        raise Exception("Min value greater than max value.")
    x_test = np.mgrid[min_:max_:grain, min_:max_:grain].reshape(2, -1).T
    if np.sqrt(x_test.shape[0]) % 2 == 0:
        size = int(np.sqrt(x_test.shape[0]))
    else:
        raise Exception('Plot topology not square!')
    return x_test, size

class Invert(object):
    """Invert matrices."""
    def __init__(self, method='inv'):
        """
        """
        if method == 'inv':
            self.inv = np.linalg.inv
        elif method == 'pinv':
            self.inv = np.linalg.pinv
        elif method == 'solve':
            self.inv = self.solve
        elif method == 'cholesky':
            self.inv = self.cholesky
        elif method == 'svd':
            self.inv = self.svd
        elif method == 'lu':
            self.inv = self.lu
        elif method == 'mp_lu':
            self.inv = self.mp_lu
        else:
            raise Exception('Invalid inversion method argument.')

    def __call__(self, arr):
        if not is_invertible(arr):
            warnings.warn('Matrix has high condition. Inverting matrix may result in errors.')
        return self.inv(arr)

    def solve(self, arr):
        identity = np.identity(arr.shape[-1], dtype=arr.dtype)
        return np.linalg.solve(arr, identity)

    def cholesky(self, arr):
        inv_cholesky = np.linalg.inv(np.linalg.cholesky(arr))
        return np.dot(inv_cholesky.T, inv_cholesky)

    def svd(self, arr):
        unitary_u, singular_values, unitary_v = np.linalg.svd(arr)
        return np.dot(unitary_v.T, np.dot(np.diag(singular_values**-1), unitary_u.T))

    def lu(self, arr):
        permutation, lower, upper = la.lu(arr)
        inv_u = np.linalg.inv(upper)
        inv_l = np.linalg.inv(lower)
        inv_p = np.linalg.inv(permutation)
        return inv_u.dot(inv_l).dot(inv_p)

    def mp_lu(self, arr):
        permutation, lower, upper = la.lu(arr)
        results = pool.map(np.linalg.inv,[upper, lower, permutation])
        # arrays of equal dimension so
        # multi_dot might be overkill
        #return np.linalg.multi_dot(results)
        return np.linalg.multi_dot(results)

def onehot(arr, num_classes=None, safe=True):
    """
    Function to take in a 1D label array and returns the one hot encoded
    transformation.
    """
    arr = atmost_1d(arr)
    if num_classes is None:
        num_classes = np.unique(arr).shape[0]
    if safe:
        if num_classes != np.unique(arr).shape[0]:
            raise Exception('Number of unique values does not match num_classes argument.')
    return np.squeeze(np.eye(num_classes)[arr.reshape(-1)])

def reversehot(arr):
    """
    Function to reverse the one hot transformation.
    """
    if len(arr.shape) > 1:
        if len(arr.shape) == 2:
            if arr.shape[0] == 1:
                return arr[0, :]
            if arr.shape[1] == 1:
                return arr[:, 0]
        return arr.argmax(axis=1)
    return arr

def deprecated(func):
    """
    A decorator used to mark functions that are deprecated with a warning.
    """
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        # https://stackoverflow.com/questions/2536307/decorators-in-the-python-standard-lib-deprecated-specifically
        # may not want to turn filter on and off
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func

# keep worker function in utils rather than
# kernel_base_multiprocessing since
# multiprocessing throws AttributeError
# if worker function in the same file as
# multiprocessing module when imported
# I know...it's weird https://bugs.python.org/issue25053
def worker(i, alpha_element, beta, m_len, distance_function):
    """
    Worker function for kernel_base_multiprocessing.
    """
    output = np.full(m_len, 0.0)
    for j in range(m_len):
        output[j] = distance_function(alpha_element, beta[j])
    return i, output.reshape(-1)
