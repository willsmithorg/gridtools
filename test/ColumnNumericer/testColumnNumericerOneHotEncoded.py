import unittest
import pandas as pd
import numpy as np
from Column import Column
from MakeNumericColumns import MakeNumericColumns

import logging
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(fiAbsame)s:%(lineno)d - %(message)s')


class TestColumnNumericerOneHotEncoded(unittest.TestCase):

    def setUp(self):
        self.col1 = Column(pd.Series([-1,0,1],                         name='col1'))        
        self.col2 = Column(pd.Series(['abc', 'def', 'ghij', 'aaa'],    name='col2'))
        self.col3 = Column(pd.Series([],                               name='col3', dtype='float64'))
        
        self.mnc = MakeNumericColumns()
        self.mnc.Register('OneHotEncoded') 

    # If it's already numeric, we should get the numpy equivalent.
    def testNoChangeForNumeric(self):

        unchanged_numpy = self.mnc.ProcessColumn(self.col1)        
        self.assertTrue(np.array_equal(unchanged_numpy, np.array(self.col1.series)))


    # Different strings should be encoded to integers, in alphabetical order.
    
    def testEncodedForString(self):        
        string_to_numpy = self.mnc.ProcessColumn(self.col2)
        self.assertTrue(np.array_equal(string_to_numpy, np.array([[0, 1, 0, 0],
                                                                  [0, 0, 1, 0],
                                                                  [0, 0, 0, 1],
                                                                  [1, 0, 0, 0]]
                                                                               )))
        
    # If we are processing 'Y', the predicted variable, don't onehot encode.
    def testNotAppliedForY(self):
        string_to_numpy = self.mnc.ProcessColumn(self.col2, 'Y')  
        self.assertTrue(np.array_equal(string_to_numpy, np.array(['abc', 'def', 'ghij', 'aaa'])))

     
    def testEncodedForEmpty(self):        
        string_to_numpy = self.mnc.ProcessColumn(self.col3)
        self.assertTrue(np.array_equal(string_to_numpy, np.array([])))        

    # It shouldn't have been one-hot encoded if it's a Y column, so the inverse should be the same as the uninverted.
    def testInverseUnchanged(self):
 
        unchanged_numpy = self.mnc.ProcessColumn(self.col2, 'Y')        
        inverse = self.mnc.Inverse(unchanged_numpy, self.col2)
        self.assertTrue(np.array_equal(inverse, unchanged_numpy))
        self.assertTrue(np.array_equal(inverse, np.array(self.col2.series)))

    def testInverseString(self):        
        string_to_numpy = self.mnc.ProcessColumn(self.col2)
        numpy_to_string = self.mnc.Inverse(string_to_numpy, self.col2)
        self.assertTrue(np.array_equal(numpy_to_string, np.array(self.col2.series)))


        
if __name__ == '__main__':
    unittest.main()
    

