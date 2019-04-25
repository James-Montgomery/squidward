import numpy as np

def make_grid(coordinates=(-10, 10, 1)):
    """
    Returns a square grid of points determined by the input coordinates
    using nump mgrid. Used in visualization fucntions.
    """
    min_, max_, grain = coordinates
    if min_ >= max_:
        raise Exception("Min value greater than max value.")
    x_test = np.mgrid[min_:max_:grain, min_:max_:grain].reshape(2, -1).T
    if np.sqrt(x_test.shape[0]) % 2 == 0:
        size = int(np.sqrt(x_test.shape[0]))
    else:
        raise Exception('Plot topology not square!')
    return x_test, size
