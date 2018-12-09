import numpy as np
import scipy as sp

def make_grid(coordinates=(-10,10,1)):
    min,max,grain = coordinates
    x = np.mgrid[min:max:grain,min:max:grain].reshape(2,-1).T
    if np.sqrt(x.shape[0]) % 2 == 0:
        s = int(np.sqrt(x.shape[0]))
    else:
        raise ValueError('Plot topology not square!')
    return x,s

def invert(A,method='inv'):
    if method == 'inv':
        return np.linalg.inv(A)
    elif method == 'pinv':
        return np.linalg.pinv(A)
    elif method == 'solve':
        I = np.identity(A.shape[-1],dtype=A.dtype)
        return np.linalg.solve(A,I)
    elif method == 'cholesky':
        c = np.linalg.inv(np.linalg.cholesky(A))
        return np.dot(c.T,c)
    elif method == 'svd':
        u,s,v = np.linalg.svd(A)
        return np.dot(v.transpose(),np.dot(np.diag(s**-1),u.transpose()))
