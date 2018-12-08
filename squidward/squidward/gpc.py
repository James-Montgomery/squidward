import numpy as np
import scipy as sp
import squidward.squidward.gpr as gpr
#from gp_viz import plot_gp

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
        self.predictors = []
        self.n_classes = None

    def preprocess(self,x,y=None):
        if len(x.shape) == 1:
            x = x.reshape(-1,1)
        if y is not None:
            y = y.reshape(-1,1)
        return x,y

    def sigmoid(self,z):
        return 1.0 / (1.0 + np.exp(-z))

    def softmax(self,predictions):
        return predictions / predictions.sum(axis=1).reshape(-1,1)

    def fit(self,x,y):
        self.x, self.y = self.preprocess(x,y)
        self.n_classes = np.unique(self.y).shape[0]
        for i in range(self.n_classes):
            y_train = np.where(self.y==i,1,-1)
            model = gpr.gaussian_process(kernel=self.kernel,var_l=self.var_l,inv_method=self.inv_method)
            model.fit(x,y_train)
            self.predictors.append(model)
        return None

    def posterior_predict(self,x_test,logits=False):
        #x_test = self.preprocess(x_test)
        means = []
        vars = []
        for model in self.predictors:
            mean, var = model.posterior_predict(x_test)
            means.append(mean)
            vars.append(var)
        if logits == False:
            means = np.array(means).T[0]
            means = self.softmax(self.sigmoid(means))
            return means.argmax(axis=1)
        else:
            means = np.array(means).T[0]
            vars = np.array(vars).T[0]
            return means, vars
