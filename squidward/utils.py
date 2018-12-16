import sys
import warnings
import numpy as np
import scipy.linalg as la

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def softmax(z):
    return z / z.sum(axis=1).reshape(-1, 1)

def is_invertible(a, strict=False):
    if strict:
        if np.linalg.cond(a) < 1.0/sys.float_info.epsilon:
            return False
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]

def check_valid_cov(cov):
    var = np.diag(cov)
    if var[var < 0].shape[0] != 0:
        raise ValueError('Negative values in diagonal of covariance matrix.\nLikely cause is kernel inversion instability.\nCheck kernel variance.')
    return None

def atleast_2d(x):
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    return x

def atmost_1d(x):
    if len(x.shape) != 1:
        return x[:, 0]
    return x

def make_grid(coordinates=(-10, 10, 1)):
    min_, max_, grain = coordinates
    x_test = np.mgrid[min_:max_:grain, min_:max_:grain].reshape(2, -1).T
    if np.sqrt(x_test.shape[0]) % 2 == 0:
        size = int(np.sqrt(x_test.shape[0]))
    else:
        raise ValueError('Plot topology not square!')
    return x_test, size

def invert(Arr, method='inv'):
    if not is_invertible(Arr):
        warnings.warn('Matrix is of low rank. Matrix might not be invertible. Recommend using LU decomposition for inversion.')
    if method == 'inv':
        return np.linalg.inv(Arr)
    elif method == 'pinv':
        return np.linalg.pinv(Arr)
    elif method == 'solve':
        I = np.identity(Arr.shape[-1], dtype=Arr.dtype)
        return np.linalg.solve(Arr, I)
    elif method == 'cholesky':
        c = np.linalg.inv(np.linalg.cholesky(Arr))
        return np.dot(c.T, c)
    elif method == 'svd':
        u, s, v = np.linalg.svd(Arr)
        return np.dot(v.transpose(), np.dot(np.diag(s**-1), u.transpose()))
    elif method == 'lu':
        P, L, U = la.lu(Arr)
        invU = np.linalg.inv(U)
        invL = np.linalg.inv(L)
        invP = np.linalg.inv(P)
        return invU.dot(invL).dot(invP)
    raise ValueError('Invalid inversion method argument.')
