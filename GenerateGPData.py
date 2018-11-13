import numpy as np
import pandas as pd

N = 10
sigma = .2
X_train = np.linspace(1,5,N).reshape(-1,1)
X_train = np.concatenate(5*[X_train])
Y_train = np.sin(5*X_train) + np.cos(.7*X_train) + np.random.normal(0,sigma,X_train.shape[0]).reshape(-1,1)
n = 500
X_test = np.linspace(0,6,n).reshape(-1,1)
Y_test = np.sin(X_test) + np.cos(.7*X_test) + np.random.normal(0,sigma,X_test.shape[0]).reshape(-1,1)

data_dict = {}
data_dict['X'] = np.concatenate([X_train,X_test])[:,0]
data_dict['Y'] = np.concatenate([Y_train,Y_test])[:,0]
data_dict['test_vs_train'] = X_train.shape[0]*['train'] + X_test.shape[0]*['test']
df = pd.DataFrame(data_dict)
df.to_csv('~/Desktop/data.csv')
