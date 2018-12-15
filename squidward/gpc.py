import numpy as np
import scipy as sp
import squidward.gpr as gpr
from squidward.utils import atleast_2d, sigmoid, softmax

np.seterr(over='raise')

class gaussian_process(object):
    def __init__(self, kernel=None, var_l=1e-15, inv_method='inv'):
        '''
        Description
        ----------
        Model object for one vs all implementation of gaussian process classification.

        Parameters
        ----------
        kernel : kernel object
            An object with an associated function k that takes in 2 arrays and
            returns a valid K matrix. Valid K matricies are positive
            semi-definite and not singular.
        var_l: float
            The liklihood variance of the process. Currently only supports
            scalars for homoskedastic regression.
        inv_method: string
            A string argument choosing an inversion method for matrix K when
            fitting the gaussian process.

        Returns
        ----------
        Model object
        '''
        self.var_l = var_l
        self.x = None
        self.y = None
        self.K = None
        self.kernel = kernel
        self.inv_method = inv_method
        self.predictors = []
        self.n_classes = None
        self.fitted = False
        assert(kernel == None, 'Model object must be instantiated with a valid kernel object.')
        assert(var_l >= 0.0, 'Invalid likelihood variance argument.')

    def fit(self, x, y):
        '''
        Description
        ----------
        Fit the model to data. This function takes in training data
        (x: features, y: targets/classes) and fits the K matrix to that data. The
        predict function can then be used to make predictions.

        Parameters
        ----------
        x: array_like
            An array containing the model features.
        y: array_like
            An array containing the model targets. Targets should be classes
            counting up from a zero idnex using integers.
            (i.e. y = [0,1,2,0,2,...])

        Returns
        ----------
        None
        '''
        self.x = atleast_2d(x)
        self.y = atleast_2d(y)
        self.n_classes = np.unique(self.y).shape[0]
        for i in range(self.n_classes):
            y_train = np.where(self.y==i, 1, -1)
            model = gpr.gaussian_process(kernel=self.kernel, var_l=self.var_l, inv_method=self.inv_method)
            model.fit(x, y_train)
            self.predictors.append(model)
        self.fitted = True
        return None

    def posterior_predict(self, x_test, logits=False):
        '''
        Description
        ----------
        Make predictions based on fitted model. This function takes in a set of
        test points to make predictions on and returns the mean function of the
        gaussian process and a measure of uncertainty (either covariance or
        variance).

        Parameters
        ----------
        x_test: array_like
            Feature input for points to make predictions for.
        logits: boolean
            If True, will return the means and variances of the one vs. all
            gaussian processes for each class. If False, returns the softmax
            class probabilities of the classes.

        Returns
        ----------
        Softmax Prob: array_like
            The softmax probabilities of each class for every test sample.
        Means: array_like
            The means of each one vs. all gaussian process for each class.
        Var: array_like
            The variance around the mean of each one vs. all gaussian process
        '''
        if self.fitted == False:
            raise ValueError('Please fit the model before trying to make posterior predictions!')
        x_test = atleast_2d(x_test)
        means = []
        vars = []
        for model in self.predictors:
            mean, var = model.posterior_predict(x_test)
            means.append(mean)
            vars.append(var)
        if logits == False:
            means = np.array(means).T[0]
            means = softmax(sigmoid(means))
            return atleast_2d(means.argmax(axis=1))
        means = np.array(means).T[0]
        vars = np.array(vars).T[0]
        return means, vars

    def prior_predict():
        '''
        While each regressor in the one vs. all gaussian process classifier has
        a prior. The softmax over their collective prior has no actual
        interpretarion and is not supported by this package.
        '''
        raise NotImplementedError('Priors not supported for One vs. All gaussian process classification.')

    def posterior_sample(self, x_test, logits=False):
        '''
        Description
        ----------
        Make predictions based on samples from the posterior of the fitted
        model. This function takes in a set of test points to make predictions
        on and returns the mean function of the gaussian process and a measure
        of uncertainty (either covariance or variance).

        Parameters
        ----------
        x_test: array_like
            Feature input for points to make predictions for.
        logits: boolean
            If True, will return the means and variances of the one vs. all
            gaussian processes for each class. If False, returns the softmax
            class probabilities of the classes.

        Returns
        ----------
        Softmax Prob: array_like
            The softmax probabilities of each class for every test sample.
        Means: array_like
            The means of each one vs. all gaussian process for each class.
        Var: array_like
            The variance around the mean of each one vs. all gaussian process
        '''
        if self.fitted == False:
            raise ValueError('Please fit the model before trying to make posterior predictions!')
        x_test = atleast_2d(x_test)
        means = []
        vars = []
        for model in self.predictors:
            mean, var = model.posterior_sample(x_test)
            means.append(mean)
            vars.append(var)
        if logits == False:
            means = np.array(means).T[0]
            means = softmax(sigmoid(means))
            return atleast_2d(means.argmax(axis=1))
        means = np.array(means).T[0]
        vars = np.array(vars).T[0]
        return means, vars

        def prior_sample(self, x_test, logits=False):
            '''
            Description
            ----------
            Make predictions based on samples from the prior of the unfitted
            model. This function takes in a set of test points to make predictions
            on and returns the mean function of the gaussian process and a measure
            of uncertainty (either covariance or variance).

            Parameters
            ----------
            x_test: array_like
                Feature input for points to make predictions for.
            logits: boolean
                If True, will return the means and variances of the one vs. all
                gaussian processes for each class. If False, returns the softmax
                class probabilities of the classes.

            Returns
            ----------
            Softmax Prob: array_like
                The softmax probabilities of each class for every test sample.
            Means: array_like
                The means of each one vs. all gaussian process for each class.
            Var: array_like
                The variance around the mean of each one vs. all gaussian process
            '''
            if self.fitted == False:
                raise ValueError('Please fit the model before trying to make posterior predictions!')
            x_test = atleast_2d(x_test)
            means = []
            vars = []
            for model in self.predictors:
                mean, var = model.prior_sample(x_test)
                means.append(mean)
                vars.append(var)
            if logits == False:
                means = np.array(means).T[0]
                means = softmax(sigmoid(means))
                return atleast_2d(means.argmax(axis=1))
            means = np.array(means).T[0]
            vars = np.array(vars).T[0]
            return means, vars
