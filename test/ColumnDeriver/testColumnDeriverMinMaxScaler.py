import unittest
import pandas as pd
import numpy as np
from Column import Column
from AddDerivedColumns import AddDerivedColumns

import logging
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(fiMinMaxScalerame)s:%(lineno)d - %(message)s')


class TestColumnDeriverMinMaxScaler(unittest.TestCase):

    def setUp(self):
        self.col1 = Column(pd.Series([-1,0,1],                         name='col1'))
        self.col2 = Column(pd.Series([-100,-3,-2,-1,0,1,2,3,+100],     name='col2'))


        
        self.col3 = Column(pd.Series(['abc', 'def', 'ghij'],           name='col3'))
 
        self.adc = AddDerivedColumns()
        self.adc.Register('MinMaxScaler') 

    # We shouldn't get a derived column on a string, or any numeric series with only all positives or all negatives.
    def testMinMaxScalerOnInvalid(self):

        newcols = self.adc.Process(self.col3)        
        self.assertEqual(len(newcols), 0)
        

    # We should get a derived column on a column with positives and negatives.
    def testMinMaxScalerOnEquallyDistributed(self):
        newcols = self.adc.Process(self.col1)    
        # 1 new column got created, it's a Column, and it contains a series that's what we expect.
        self.assertEqual(len(newcols), 1)
        self.assertIsInstance(newcols[0], Column)
        self.assertEqual(newcols[0].series.name, 'col1.MinMaxScaler')   
        #print(newcols[0].series)
        self.assertTrue(np.allclose(newcols[0].series, pd.Series([0, 0.5, +1]), atol=1e-4))       
        
        # Also applies to floating points.       
        newcols = self.adc.Process(self.col2)    
        # 1 new column got created, it's a Column, and it contains a series that's what we expect.
        self.assertEqual(len(newcols), 1)
        self.assertIsInstance(newcols[0], Column)
        self.assertEqual(newcols[0].series.name, 'col2.MinMaxScaler')    
        #print(newcols[0].series)        
        self.assertTrue(np.allclose(newcols[0].series, pd.Series([0, 0.485, 0.49, 0.495, 0.50, 0.505, 0.510, 0.515, 1.000]), atol=1e-3))
                
        
if __name__ == '__main__':
    unittest.main()
    

