from squidward import utils
import numpy as np
import unittest

class TestUtilities(unittest.TestCase):

    def test_is_invertible_true(self):
        '''
        Tests that non-singular matricies return true.
        '''
        arr = np.random.rand(10, 10)
        arr = arr.dot(arr.T)
        output = utils.is_invertible(arr)
        assert(output == True)

    def test_is_invertible_false(self):
        '''
        Tests that singular matricies return false.
        '''
        arr = np.random.rand(10, 10)
        arr[-1] = arr[0] + arr[1]
        output = utils.is_invertible(arr)
        assert(output == False)

if __name__ == '__main__':
    unittest.main()
