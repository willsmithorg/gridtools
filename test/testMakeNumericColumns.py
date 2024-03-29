import unittest
import pandas as pd
import numpy as np
from Column import Column
from MakeNumericColumns import MakeNumericColumns

import logging
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')


class TestMakeNumericColumns(unittest.TestCase):

    def setUp(self):
        self.col1 = Column(pd.Series(['abc','def', 'ghij', 'abc'], name='ser1'))
        self.col2 = Column(pd.Series([1, 2, 4],                    name='ser2'))
    
    def testInitBadParams1(self):
        # Test bad calls to the constructor.
        with self.assertRaises(TypeError):
            mnc = MakeNumericColumns(123)
 
    def testInitGood(self):
        # Check we the constructor returns something sane if passed good params.
        mnc = MakeNumericColumns()
        self.assertIsInstance(mnc, MakeNumericColumns)

    def testRegisterAndProcess(self):
        mnc = MakeNumericColumns()
        mnc.Register('LabelEncoded')

        numeric = mnc.ProcessColumn(self.col1)        
        self.assertGreater(len(numeric), 0)
        self.assertIsInstance(numeric, np.ndarray)
        self.assertEqual(numeric.shape, (4,))        

    def testMultipleEncodings(self):
        mnc = MakeNumericColumns()
        mnc.Register('OneHotEncoded')        
        mnc.Register('LabelEncoded')

        numeric = mnc.ProcessColumn(self.col1)        
        self.assertGreater(len(numeric), 0)
        self.assertIsInstance(numeric, np.ndarray)
        self.assertEqual(numeric.shape, (4, 3))         # it shuold have got one-hot encoded.  4 cells but only 3 labels needed.

    def testInverse(self):
        # Test we can encode with OneHot then invert.
        mnc = MakeNumericColumns()
        mnc.Register('OneHotEncoded')        

        numeric = mnc.ProcessColumn(self.col1)   
        inverse = mnc.Inverse(numeric, self.col1)        
        self.assertTrue(np.array_equal(inverse, np.array(self.col1.series)))

        numeric = mnc.ProcessColumn(self.col2)   
        inverse = mnc.Inverse(numeric, self.col2)        
        self.assertTrue(np.array_equal(inverse, np.array(self.col2.series)))        
        
        # Test we can encode with LabelEncoded then invert.
        mnc = MakeNumericColumns()        
        mnc.Register('LabelEncoded')        

        numeric = mnc.ProcessColumn(self.col1)   
        inverse = mnc.Inverse(numeric, self.col1)        
        self.assertTrue(np.array_equal(inverse, np.array(self.col1.series)))

        numeric = mnc.ProcessColumn(self.col2)   
        inverse = mnc.Inverse(numeric, self.col2)        
        self.assertTrue(np.array_equal(inverse, np.array(self.col2.series)))           
        
if __name__ == '__main__':
    unittest.main()
    

