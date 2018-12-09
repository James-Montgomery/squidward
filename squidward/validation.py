import numpy as np

# add posterior checks

def rmse(p,y):
    return np.sqrt(np.sum((p - y)**2)/y.shape[0])

def acc(p,y):
    if y.shape[0] <= 1:
        return y.T[y.T == p].shape[0] / y.shape[0]
    else:
        return y[y == p].shape[0] / y.shape[1]

def brier_score():
    raise NotImplementedError()

def precision():
    raise NotImplementedError()

def recall():
    raise NotImplementedError()

def roc_auc():
    raise NotImplementedError()
