import unittest
import pandas as pd
import numpy as np
from Column import Column
from AddDerivedColumns import AddDerivedColumns

import logging
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(fiAbsame)s:%(lineno)d - %(message)s')


class TestColumnDeriverAbs(unittest.TestCase):

    def setUp(self):
        self.col1 = Column(pd.Series([-1,0,1],                         name='col1'))
        self.col2 = Column(pd.Series([-1.1,0,2.1],                     name='col2'))

        self.col3 = Column(pd.Series([-1.1,0,-2.1],                     name='col3'))
        self.col4 = Column(pd.Series([+1.1,0,+2.1],                     name='col4'))
        
        self.col5 = Column(pd.Series(['abc', 'def', 'ghij'],           name='col5'))
 
        self.adc = AddDerivedColumns()
        self.adc.Register('Abs') 

    # We shouldn't get a derived column on a string, or any numeric series with only all positives or all negatives.
    def testAbsOnInvalid(self):

        newcols = self.adc.Process(self.col3)        
        self.assertEqual(len(newcols), 0)
        
        newcols = self.adc.Process(self.col4)        
        self.assertEqual(len(newcols), 0)
        
        newcols = self.adc.Process(self.col5)        
        self.assertEqual(len(newcols), 0)

    # We should get a derived column on a column with positives and negatives.
    def testAbsOnAllPositive(self):
        newcols = self.adc.Process(self.col1)    
        # 1 new column got created, it's a Column, and it contains a series that's what we expect.
        self.assertEqual(len(newcols), 1)
        self.assertIsInstance(newcols[0], Column)
        self.assertEqual(newcols[0].series.name, 'col1.Abs')        
        self.assertTrue(pd.Series.equals(newcols[0].series, pd.Series([+1, 0, +1])))       
        
        # Also applies to floating points.       
        newcols = self.adc.Process(self.col2)    
        # 1 new column got created, it's a Column, and it contains a series that's what we expect.
        self.assertEqual(len(newcols), 1)
        self.assertIsInstance(newcols[0], Column)
        self.assertEqual(newcols[0].series.name, 'col2.Abs')        
        self.assertTrue(np.allclose(newcols[0].series, pd.Series([+1.1, 0, +2.1]), atol=1e-8))
                
        
if __name__ == '__main__':
    unittest.main()
    

