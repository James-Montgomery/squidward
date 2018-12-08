import numpy as np
import scipy as sp
#from gp_viz import plot_gp

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
