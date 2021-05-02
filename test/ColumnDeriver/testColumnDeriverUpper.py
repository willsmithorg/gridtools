import unittest
import pandas as pd
import numpy as np
from Column import Column
from AddDerivedColumns import AddDerivedColumns

import logging
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')


class TestColumnDeriverUpper(unittest.TestCase):

    def setUp(self):
        self.col1 = Column(pd.Series(['abc','def', 'ghij', 'ABC'], name='col1'))
        self.col2 = Column(pd.Series([1, 2, 4],                    name='col1'))
 
        self.adc = AddDerivedColumns()
        self.adc.Register('Upper') 

    # We shouldn't get a derived column on a numeric.
    def testUpperOnNumeric(self):

        newcols = self.adc.Process(self.col2)        
        self.assertEqual(len(newcols), 0)

    # We should get a derived column on a string column.
    def testLenOnString(self):
        newcols = self.adc.Process(self.col1)    
        # 1 new column got created, it's a Column, and it contains a series that's what we expect.
        self.assertEqual(len(newcols), 1)
        self.assertIsInstance(newcols[0], Column)
        self.assertEqual(newcols[0].series.name, 'col1.Upper')                
        self.assertTrue(pd.Series.equals(newcols[0].series, pd.Series(['ABC', 'DEF', 'GHIJ', 'ABC'])))
        
        
if __name__ == '__main__':
    unittest.main()
    

