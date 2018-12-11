from squidward import utils
import numpy as np

def test_is_invertible_true():
    '''
    Tests that non-singular matricies return true.
    '''
    arr = np.random.rand(10, 10)
    arr = arr.dot(arr.T)
    output = utils.is_invertible(arr)
    assert(output == True)

def test_is_invertible_false():
    '''
    Tests that singular matricies return false.
    '''
    arr = np.random.rand(10, 10)
    arr[-1] = arr[0] + arr[1]
    output = utils.is_invertible(arr)
    assert(output == False)
