import unittest
import pandas as pd
import numpy as np
from Column import Column
from AddDerivedColumns import AddDerivedColumns

import logging
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')


class TestColumnDeriverDate(unittest.TestCase):

    def setUp(self):
        self.col1 = Column(pd.Series(['20200101', '20200320', 'contains 20210116',  '20201231', 'bad',
                                 '20-01-01', '99-03-20', '51-01-16',  '2130-12-31'  ],      name='col1'))

        self.col2 = Column(pd.Series([1, 2, 4],                                             name='col2'))
        self.col3 = Column(pd.Series(['abc', 'def', 'ghij', '20010101'],                    name='col3'))

 
        self.adc = AddDerivedColumns()
        self.adc.Register('Date') 

    # We shouldn't get a derived column on a numeric.
    def testDateOnNumeric(self):

        newcols = self.adc.Process(self.col2)        
        self.assertEqual(len(newcols), 0)

    # We shouldn't get a derived column on a string without many dates.
    def testDateOnStrings(self):
        newcols = self.adc.Process(self.col3)        
        self.assertEqual(len(newcols), 0)
        
        
    # We should get a derived column on a string column if most of them contain dates.
    def testLenOnString(self):
        newcols = self.adc.Process(self.col1)    
        # 3 new columns, for year+month+date.
        self.assertEqual(len(newcols), 3)
        self.assertIsInstance(newcols[0], Column)
        self.assertIsInstance(newcols[1], Column)
        self.assertIsInstance(newcols[2], Column)
        
        self.assertEqual(newcols[0].series.name, 'col1.Date_year')  
        self.assertEqual(newcols[1].series.name, 'col1.Date_month')                
        self.assertEqual(newcols[2].series.name, 'col1.Date_day')                
        
        self.assertTrue(pd.Series.equals(newcols[0].series,  pd.Series([2020.0, 2020.0, 2021.0, 2020.0,  np.nan, 2020.0, 1999.0, 2051.0, 2130.0 ])))
        self.assertTrue(pd.Series.equals(newcols[1].series, pd.Series([   1.0,    3.0,    1.0,   12.0,   np.nan,    1.0,    3.0,    1.0,   12.0 ])))
        self.assertTrue(pd.Series.equals(newcols[2].series,   pd.Series([   1.0,   20.0,   16.0,   31.0, np.nan,    1.0,   20.0,   16.0,   31.0 ])))
        
        
if __name__ == '__main__':
    unittest.main()
    

