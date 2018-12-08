import numpy as np
import scipy as sp
from squidward.squidward.utils import invert
#from squidward.squidward import gp_viz

np.seterr(over='raise')

def check_valid_cov(cov):
    var = np.diag(cov)
    if var[var<0].shape[0] != 0:
        raise ValueError('Negative values in diagonal of covariance matrix.\nLikely cause is kernel inversion instability. Check kernel variance.')
    else:
        pass

class gaussian_process(object):
    def __init__(self,kernel=None,var_l=1e-15,inv_method='inv'):
        '''
        '''
        self.var_l = var_l
        self.x = None
        self.y = None
        self.K = None
        self.kernel = kernel
        self.inv_method = inv_method

    def fit(self,x,y):
        self.x = x
        self.y = y
        K = self.kernel.k(x,x)

        I = np.zeros(K.shape)
        idx = np.diag_indices(I.shape[0])
        I[idx] = self.var_l + 1e-15 # prevent numerical instability
        K += I

        #K += np.eye(K.shape)*self.var_l
        self.K = invert(K,self.inv_method)

    def posterior_predict(self,x_test,return_cov=False):
        K_s = self.kernel.k(x_test,self.x)
        mean = K_s.dot(self.K).dot(self.y)
        K_ss = self.kernel.k(x_test,x_test)
        cov = K_ss - np.dot(np.dot(K_s,self.K),K_s.T)
        check_valid_cov(cov)
        if return_cov == True:
            return mean, cov
        else:
            return mean, np.diag(cov).reshape(-1,1) #variance

    def prior_predict(self,x_test,return_cov=False):
        mean = np.zero(x_test.shape[0])
        cov = self.kernel.k(x_test,x_test)
        check_valid_cov(cov)
        if return_cov == True:
            return mean, cov
        else:
            return mean, np.diag(cov).reshape(-1,1) #variance

    def posterior_sample(self,x_test):
        mean,cov = posterior_predict(x_test,True)
        return np.random.multivariate_normal(mean,cov,1).T[:,0]

    def prior_sample(self,x_test):
        mean,cov = prior_predict(x_test,True)
        return np.random.multivariate_normal(mean,cov,1).T[:,0]

    def validate(self,x,mean,var):
        raise NotImplementedError()

def variational_inference():
    raise NotImplementedError()

def MCMC():
    raise NotImplementedError()

def bayes_nets_approximation():
    raise NotImplementedError()
