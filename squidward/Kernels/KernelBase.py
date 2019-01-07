import multiprocessing
import numpy as np

class Kernel(object):
    """
    """
    def __init__(self, distance_function, method='k1', pool_size=1):
        """
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
        n, m = alpha.shape[0], beta.shape[0]
        cov = np.full((n,m),0.0)
        for i in range(n):
            for j in range(m):
                cov[i,j] = self.distance_function(alpha[i],beta[j])
        return cov
