import unittest
import pandas as pd
import numpy as np
from Column import Column
from AddDerivedColumns import AddDerivedColumns

import logging
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')


class TestColumn(unittest.TestCase):

    def setUp(self):
        self.col1 = Column(pd.Series(['abc','def', 'ghij', 'abc'], name='ser1'))
        self.col2 = Column(pd.Series([1, 2, 4],                    name='ser2'))
    
    def testInitBadParams1(self):
        # Test bad calls to the constructor.
        with self.assertRaises(TypeError):
            adc = AddDerivedColumns(123)
 
    def testInitGood(self):
        # Check we the constructor returns something sane if passed good params.
        adc = AddDerivedColumns()
        self.assertIsInstance(adc, AddDerivedColumns)

    def testRegister(self):
        adc = AddDerivedColumns()
        adc.Register('Len')

        newcols = adc.Process(self.col1)        
        self.assertGreater(len(newcols), 0)
        self.assertIsInstance(newcols[0], Column)



if __name__ == '__main__':
    unittest.main()
    

