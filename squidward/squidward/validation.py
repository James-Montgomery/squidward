import numpy as np

# add posterior checks

def rmse(p,y):
    return np.sqrt(np.sum((p - y)**2)/y.shape[0])
