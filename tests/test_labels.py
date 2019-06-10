'''
Created on 10.06.2019

@author: Philipp
'''
import unittest
from wildlife.dataset.wildlife.labels import convert_label_to_ids


class Test(unittest.TestCase):

    def convert_label_to_ids(self):
        label_to_id = {'hare': 6, 'marten': 5, 'background': 0, 'horse': 1, 'wildboar': 8, 'dog': 3, 'fox': 10, 'cat': 2, 'deer': 4, 'bird': 7, 'racoon': 9}
        """
            Convert a y_dataset that is based on label names to corresponding id numbers.
            For example: (deer, tree, human) becomes (1, 0, 0) for {b'tree': 0, b'human':0, b'deer':1}
        """
        y_dataset = [b'deer', b'background', b'background', b'background', b'background']
        convert_label_to_ids(y_dataset, label_to_id)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
