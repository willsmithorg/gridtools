import unittest
import pandas as pd
import numpy as np
from SyntheticError import SyntheticError

import logging
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')


class SyntheticError(unittest.TestCase):

    def setUp(self):
        pass
        
    def testInitBadParams1(self):
        # Test bad calls to the constructor.
        with self.assertRaises(TypeError):
            se = SyntheticError(123)
 
    def testInitGood(self):
        # Check we the constructor returns something sane if passed good params.
        se = SyntheticError()
        self.assertIsInstance(se, SyntheticError)

    def TestMakeClassificationDefaults(self):
    
        se = SyntheticError()
        df = se.MakeClassification()
        self.assertIsInstance(pd.DataFrame)
        
        
if __name__ == '__main__':
    unittest.main()
    

