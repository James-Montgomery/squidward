import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns

# One Dimensional Plots

def plot_1dr(x,mean,var):
    plt.fill_between(x,
                 mean-.674*np.sqrt(var),
                 mean+.674*np.sqrt(var),
                 color='k',alpha=.4,label='50% Credible Interval')
    plt.fill_between(x,
                 mean-1.150*np.sqrt(var),
                 mean+1.150*np.sqrt(var),
                 color='k',alpha=.3,label='75% Credible Interval')
    plt.fill_between(x,
                 mean-1.96*np.sqrt(var),
                 mean+1.96*np.sqrt(var),
                 color='k',alpha=.2,label='95% Credible Interval')
    plt.fill_between(x,
                 mean-2.326*np.sqrt(var),
                 mean+2.326*np.sqrt(var),
                 color='k',alpha=.1,label='99% Credible Interval')
    plt.plot(x,mean,c='w')
    return None

def plot_1dc(x,mean):
    plt.plot(x,mean,c='w')
    return None

# Two Dimensional Plots

def plot_2dr(model,contour=True,coordinates=(-10,10,1)):
    min,max,grain = coordinates
    x_test = np.mgrid[min:max:grain,min:max:grain].reshape(2,-1).T
    mean, var = model.posterior_predict(x_test)

    if np.sqrt(x_test.shape[0]) % 2 == 0:
        s = int(np.sqrt(x_test.shape[0]))
    else:
        raise ValueError('Plot topology not square!')

    if contour == True:
        plt.figure(figsize=(20,10))
        plt.subplot(2,2,1)
        plt.scatter(x_test[:,0],x_test[:,1],c=mean)

        plt.subplot(2,2,2)
        plt.scatter(x_test[:,0],mean)
        plt.scatter(x_test[:,1],mean)

        plt.subplot(2,2,3)
        a,b = x_test.T.reshape(2,s,s)
        z = mean.T.reshape(s,s)
        plt.contourf(a, b, z, 20, cmap='RdGy')
        plt.colorbar()
        contours = plt.contour(a, b, z, 5, colors='black')
        plt.clabel(contours, inline=True, fontsize=8)

        plt.subplot(2,2,4)
        a,b = x_test.T.reshape(2,s,s)
        z = np.sqrt(var).T.reshape(s,s)
        plt.contourf(a, b, z, 20, cmap='RdGy')
        plt.colorbar()
        contours = plt.contour(a, b, z, 5, colors='black')
        plt.clabel(contours, inline=True, fontsize=8)
    else:
        fig = plt.figure(figsize=(20,10))

        ax = fig.add_subplot(221, projection="3d")
        a,b = x_test.T.reshape(2,s,s)
        z = mean.T.reshape(s,s)
        ax.plot_surface(a, b, z, cmap="autumn_r", lw=0.5, rstride=1, cstride=1, alpha=0.5)
        ax.contour(a, b, z, 10, lw=3, colors="k", linestyles="solid")
        ax.view_init(30, -60)

        ax = fig.add_subplot(222, projection="3d")
        ax.plot_surface(a, b, z, cmap="autumn_r", lw=0.5, rstride=1, cstride=1, alpha=0.5)
        ax.contour(a, b, z, 10, lw=3, colors="k", linestyles="solid")
        ax.view_init(30, 60)

        ax = fig.add_subplot(223, projection="3d")
        a,b = x_test.T.reshape(2,s,s)
        z = np.sqrt(var).T.reshape(s,s)
        ax.plot_surface(a, b, z, cmap="autumn_r", lw=0.5, rstride=1, cstride=1, alpha=0.5)
        ax.contour(a, b, z, 10, lw=3, colors="k", linestyles="solid")
        ax.view_init(30, -60)

        ax = fig.add_subplot(224, projection="3d")
        ax.plot_surface(a, b, z, cmap="autumn_r", lw=0.5, rstride=1, cstride=1, alpha=0.5)
        ax.contour(a, b, z, 10, lw=3, colors="k", linestyles="solid")
        ax.view_init(30, 60)
    return None

def plot_2dc(model,coordinates=(-10,10,1)):
    min,max,grain = coordinates
    x_test = np.mgrid[min:max:grain,min:max:grain].reshape(2,-1).T
    mean = model.posterior_predict(x_test)

    if np.sqrt(x_test.shape[0]) % 2 == 0:
        s = int(np.sqrt(x_test.shape[0]))
    else:
        raise ValueError('Plot topology not square!')

    plt.figure(figsize=(20,10))
    plt.subplot(1,2,1)
    plt.scatter(x_test[:,0],x_test[:,1],c=mean)

    plt.subplot(1,2,2)
    a,b = x_test.T.reshape(2,s,s)
    z = mean.T.reshape(s,s)
    plt.contourf(a, b, z, 20, cmap='RdGy')
    plt.colorbar()
    contours = plt.contour(a, b, z, 5, colors='black')
    plt.clabel(contours, inline=True, fontsize=8)

    return None

def test(model,coordinates=(-10,10,1)):
    min,max,grain = coordinates
    x_test = np.mgrid[min:max:grain,min:max:grain].reshape(2,-1).T
    mean, var = model.posterior_predict(x_test,True)
    var = np.min(var,axis=1)

    if np.sqrt(x_test.shape[0]) % 2 == 0:
        s = int(np.sqrt(x_test.shape[0]))
    else:
        raise ValueError('Plot topology not square!')

    plt.figure(figsize=(20,10))
    a,b = x_test.T.reshape(2,s,s)
    z = var.T.reshape(s,s)
    plt.contourf(a, b, z, 100, cmap='RdGy')
    plt.colorbar()
    contours = plt.contour(a, b, z, 5, colors='black')
    plt.clabel(contours, inline=True, fontsize=8)

    return None
