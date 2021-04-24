import unittest
import pandas as pd
import numpy as np
from Explain import Explain

import logging
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')

class TestSpotErrors(unittest.TestCase):


    def setUp(self):    

        # Mixture of categorical and non-categorical data.
        data5 = { 'country': ['Germany','Germany','Germany','Germany','Germany', 'Germany',
                                     'US', 'US', 'US', 'US', 'US',
                                     'UK', 'UK', 'UK', 'UK', 'UK', 
                                     'LUX','LUX','LUX','LUX','LUX'],
                         'currency':['EUR','EUR','EUR','EUR','EUR','MXN',
                                     'USD','USD','USD','USD','USD',
                                     'GBP','GBP','GBP','GBP','GBP',
                                     'EUR','EUR','EUR','EUR','EUR'],
                         'manyvalues':['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E', 'E', 'E', 'K', 'L', 'M', 'N', 'O',
                                      'Z', 'Z', 'Z', 'Z', 'Y'],
                         'rand': [ np.random.uniform(100,200) for _ in range(21) ],
                         'linear': [ x for x in range(21) ]}
        self.data5 = pd.DataFrame(data5)        
        self.data5_count = len(self.data5['country'])                         
         

    def testInitGood(self):    
        e = Explain()
        self.assertIsInstance(e, Explain)
        
    def testCalcPercentageOfRows(self):
        e = Explain()
        (t, m1, m2) = e._CalcPercentageOfRows(self.data5, 'manyvalues', ['currency', 'country'], 20, 'Y', 'Z')
        
        self.assertEqual(t, 5)
        self.assertEqual(m1, 1)
        self.assertEqual(m2, 4)
        
        (t, m1, m2) = e._CalcPercentageOfRows(self.data5, 'manyvalues', ['currency', 'country'], 20, 'Y', None)
        
        self.assertEqual(t, 5)
        self.assertEqual(m1, 1)
        self.assertEqual(m2, 0)
        
        
if __name__ == '__main__':
    unittest.main()
    
