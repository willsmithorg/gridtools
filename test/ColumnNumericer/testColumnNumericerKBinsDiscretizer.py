import unittest
import pandas as pd
import numpy as np
from Column import Column
from MakeNumericColumns import MakeNumericColumns

import logging
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(fiAbsame)s:%(lineno)d - %(message)s')


class TestColumnNumericerKBinsDiscretizer(unittest.TestCase):

    def setUp(self):
        self.col1 = Column(pd.Series([-1,0,1],                         name='col1'))        
        self.col2 = Column(pd.Series(np.linspace(1,5,100),             name='col2'))
        self.col3 = Column(pd.Series([],                               name='col3', dtype='float64'))
        
        self.mnc = MakeNumericColumns()
        self.mnc.Register('KBinsDiscretizer') 

    # If it's already numeric, we should get the numpy equivalent.
    def testNotAppliedForLowNunique(self):

        unchanged_numpy = self.mnc.ProcessColumn(self.col1)        
        self.assertTrue(np.array_equal(unchanged_numpy, np.array(self.col1.series)))


       
    # If we are processing 'X', don't encode.  We only want to encode Y so everything becomes a classifier.
    def testNotAppliedForX(self):
        processed = self.mnc.ProcessColumn(self.col2, 'X')  
        self.assertTrue(np.array_equal(processed, np.linspace(1,5,100)))

     
    def testEncodedForEmpty(self):        
        processed = self.mnc.ProcessColumn(self.col3)
        self.assertTrue(np.array_equal(processed, np.array([])))      

    # It shouldn't have been converted if it's an X column, so the inverse should be the same as the uninverted.
    def testInverseUnchanged(self):
 
        unchanged_numpy = self.mnc.ProcessColumn(self.col2, 'X')        
        inverse = self.mnc.Inverse(unchanged_numpy, self.col2, 'X')
        self.assertTrue(np.array_equal(inverse, unchanged_numpy))
        self.assertTrue(np.array_equal(inverse, np.array(self.col2.series)))

    def testInverse(self):        
        discretized = self.mnc.ProcessColumn(self.col2, 'Y')
        inverse = self.mnc.Inverse(discretized, self.col2, 'Y')
        #print(np.array(self.col2.series))
        #print(inverse)
        # Due to discretization error, we expect them to be within about 0.1 of each other.  
        # So definitely within 0.2 but not all within 0.05
        self.assertFalse(np.array_equal(inverse, np.array(self.col2.series)))        
        self.assertFalse(np.allclose(inverse, np.array(self.col2.series), atol=0.05))
        self.assertTrue(np.allclose(inverse, np.array(self.col2.series), atol=0.2))

        
if __name__ == '__main__':
    unittest.main()
    

