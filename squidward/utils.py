"""
This script contains code for useful data transformations and data checks
used in the other modules of squidward.
"""

import sys
import warnings
import functools
import numpy as np
import scipy.linalg as la

try:
    # python 2
    numeric_types = (int, float, long, complex)
except:
    # python 3
    numeric_types = (int, float, complex)

np.seterr(over="raise")

# run np.show_config() to see
# if numpy is running with MKL
# backend

# ---------------------------------------------------------------------------------------------------------------------
# Array Checks
# ---------------------------------------------------------------------------------------------------------------------


def array_equal(alpha, beta):
    """
    Function returns true if two arrays are identical.
    """
    # if alpha.shape == beta.shape:
    #     if np.all(np.sort(alpha) == np.sort(beta):
    #         return True
    # return False
    return np.array_equal(alpha, beta)


def exactly_1d(arr):
    """
    Function to ensure that an array has a most 1 dimension. Used to
    formalize output / input dimensions for certain functions.
    """
    if not isinstance(arr, np.ndarray):
        if isinstance(arr, numeric_types):
            return np.array([arr])
        else:
            raise Exception("Not appropriate input type.")
    if len(arr.shape) == 1:
        return arr
    if len(arr.shape) == 2:
        if arr.shape[0] == 1:
            return arr[0, :]
        if arr.shape[1] == 1:
            return arr[:, 0]
    raise Exception("Not appropriate input shape.")


def exactly_2d(arr):
    """
    Function to ensure that an array has a least 2 dimensions. Used to
    formalize output / input dimensions for certain functions.
    """
    arr = np.asarray(arr)
    if len(arr.shape) == 1:
        return arr.reshape(-1, 1)
    if len(arr.shape) == 2:
        if arr.shape[0] == 1:
            return arr.reshape(-1, 1)
        return arr
    if len(arr.shape) == 3:
        if arr.shape[0] == 1:
            return arr[0, :, :]
        if arr.shape[2] == 1:
            return arr[:, :, 0]
        raise Exception("Not appropriate input shape.")
    if len(arr.shape) > 3:
        raise Exception("Not appropriate input shape.")
    raise Exception("Not appropriate input shape.")

# ---------------------------------------------------------------------------------------------------------------------
# Inversions
# ---------------------------------------------------------------------------------------------------------------------


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


def check_valid_cov(cov, safe=True):
    """
    Function to do safety checks on covariance matrices.
    """
    if not safe:
        return None
    if not is_invertible(cov):
        warnings.warn('Cov has high condition. Inverting matrix may result in errors.')
    var = np.diag(cov)
    if var[var < 0].shape[0] != 0:
        raise Exception('Negative values in diagonal of covariance matrix.\nLikely cause is kernel inversion instability.\nCheck kernel variance.')


class Invert(object):
    """Invert matrices."""
    def __init__(self, method='inv'):
        """
        Description
        ----------
        Class to handle inverting matrices.

        Parameters
        ----------
        method: String
            The name of the method to be used for inverting matrices.
            Options: inv, pinv, solve, cholesky, svd, lu, mp_lu
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
        """
        Inverts matrix.
        """
        if not is_invertible(arr):
            warnings.warn('Matrix has high condition. Inverting matrix may result in errors.')
        return self.inv(arr)

    def solve(self, arr):
        """
        Use cramer emthod for finding matrix inversion.
        """
        identity = np.identity(arr.shape[-1], dtype=arr.dtype)
        return np.linalg.solve(arr, identity)

    def cholesky(self, arr):
        """
        Use cholesky decomposition for finding matrix inversion.
        """
        inv_cholesky = np.linalg.inv(np.linalg.cholesky(arr))
        return np.dot(inv_cholesky.T, inv_cholesky)

    def svd(self, arr):
        """
        Use singular value decomposition for finidng matrix inversion.
        """
        unitary_u, singular_values, unitary_v = np.linalg.svd(arr)
        return np.dot(unitary_v.T, np.dot(np.diag(singular_values**-1), unitary_u.T))

    def lu(self, arr):
        """
        Use lower upper decomposition for finding amtrix inversion.
        """
        permutation, lower, upper = la.lu(arr)
        inv_u = np.linalg.inv(upper)
        inv_l = np.linalg.inv(lower)
        inv_p = np.linalg.inv(permutation)
        return inv_u.dot(inv_l).dot(inv_p)

# ---------------------------------------------------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------------------------------------------------


def onehot(arr, num_classes=None, safe=True):
    """
    Function to take in a 1D label array and returns the one hot encoded
    transformation.
    """
    arr = exactly_1d(arr)
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

# ---------------------------------------------------------------------------------------------------------------------
# Classification Specific
# ---------------------------------------------------------------------------------------------------------------------


def sigmoid(z):
    """
    Function to return the sigmoid transformation for every
    term in an array.
    """
    # TODO: find a better way to get this working
    np.seterr(over="warn")
    sig = 1.0 / (1.0 + np.exp(-z))
    sig = np.minimum(sig, 1.0)  # Set upper bound
    sig = np.maximum(sig, 0.0)  # Set lower bound
    np.seterr(over="raise")
    return sig


def softmax(z):
    """
    Function to return the softmax transformation over an
    input vector.
    """
    z = sigmoid(z)
    return z / z.sum(axis=1).reshape(-1, 1)

# ---------------------------------------------------------------------------------------------------------------------
# Miscellaneous
# ---------------------------------------------------------------------------------------------------------------------


def deprecated(func):
    """
    A decorator used to mark functions that are deprecated with a warning.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # https://stackoverflow.com/questions/2536307/decorators-in-the-python-standard-lib-deprecated-specifically
        # may not want to turn filter on and off
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        primary_message = "Call to deprecated function {}.".format(func.__name__)
        warnings.warn(primary_message, category=DeprecationWarning, stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return wrapper
