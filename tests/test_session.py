'''
Created on 10.06.2019

@author: Philipp
'''
import unittest
from wildlife.session import to_categorical
import numpy as np


class Test(unittest.TestCase):

    def to_categorical(self):
        label_to_id = {'hare': 6, 'marten': 5, 'background': 0, 'horse': 1, 'wildboar': 8, 'dog': 3, 'fox': 10, 'cat': 2, 'deer': 4, 'bird': 7, 'racoon': 9}
        y_dataset = [b'deer', b'background', b'background', b'background', b'background']
        _, result = to_categorical(y_dataset, label_to_id)
        # expected five entries with 11 categories (number of categories)
        assert np.shape(result) == (5, 11)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
