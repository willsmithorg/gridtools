import unittest
import pandas as pd
import numpy as np
from Column import Column
from AddDerivedColumns import AddDerivedColumns

import logging
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(fiTokenizerCharDecimalame)s:%(lineno)d - %(message)s')


class TestColumnDeriverTokenizerCharDecimal(unittest.TestCase):

    def setUp(self):
        self.col1 = Column(pd.Series([-1,0,1],                         name='col1'))
        self.col2 = Column(pd.Series(['this is a really long string and we would prefer if really long -- more than 30 character -- strings didn''t get tokenized']))
        self.col3 = Column(pd.Series(['AB101X','AB1033=Y','XY123','MYN199F3','123ABC', 'mmm.123', '999', '', 'ZZZ'],           name='col3'))
 
        self.col4 = Column(pd.Series(['', '', '']))
        
        self.adc = AddDerivedColumns()
        self.adc.Register('TokenizerCharDecimal') 

    # We shouldn't get a derived column on a string, or any numeric series with only all positives or all negatives.
    def testTokenizerCharDecimalOnInvalid(self):

        newcols = self.adc.Process(self.col1)        
        self.assertEqual(len(newcols), 0)
        newcols = self.adc.Process(self.col2)        
        self.assertEqual(len(newcols), 0)        
       

    # We should get a derived column on a column with positives and negatives.
    def testTokenizerCharDecimalOnVarious(self):
        newcols = self.adc.Process(self.col3)    
        # 1 new column got created, it's a Column, and it contains a series that's what we expect.
        self.assertEqual(len(newcols), 9)
        
        for i in range(9):
            self.assertIsInstance(newcols[i], Column)          
        
        self.assertEqual(newcols[0].series.name, 'col3.TokenizerCharDecimal_characters1')  
        self.assertEqual(newcols[1].series.name, 'col3.TokenizerCharDecimal_characters2')   
        self.assertEqual(newcols[2].series.name, 'col3.TokenizerCharDecimal_reversecharacters1')  
        self.assertEqual(newcols[3].series.name, 'col3.TokenizerCharDecimal_reversecharacters2')         
        self.assertEqual(newcols[4].series.name, 'col3.TokenizerCharDecimal_digits1')                
        self.assertEqual(newcols[5].series.name, 'col3.TokenizerCharDecimal_digits2')
        self.assertEqual(newcols[6].series.name, 'col3.TokenizerCharDecimal_reversedigits1')                
        self.assertEqual(newcols[7].series.name, 'col3.TokenizerCharDecimal_reversedigits2')        
        self.assertEqual(newcols[8].series.name, 'col3.TokenizerCharDecimal_digitsfirst')

        self.assertTrue(pd.Series.equals(newcols[0].series, pd.Series(['AB', 'AB',  'XY', 'MYN', 'ABC',     'mmm',   None,  None, 'ZZZ'])))       
        self.assertTrue(pd.Series.equals(newcols[1].series, pd.Series(['X',  'Y',   None,   'F',   None,     None,   None,  None,  None ])))     
        self.assertTrue(pd.Series.equals(newcols[2].series, pd.Series(['X', 'Y',   'XY',    'F', 'ABC',     'mmm',   None,  None, 'ZZZ' ])))       
        self.assertTrue(pd.Series.equals(newcols[3].series, pd.Series(['AB',  'AB', None, 'MYN',   None,     None,   None,  None,  None])))  
        
        
        self.assertTrue(pd.Series.equals(newcols[4].series, pd.Series(['101','1033','123','199',  '123',     '123',  '999', None,  None]))) 
        self.assertTrue(pd.Series.equals(newcols[5].series, pd.Series([None   ,None, None,   '3',   None,    None,   None , None,  None])))   
        self.assertTrue(pd.Series.equals(newcols[6].series, pd.Series(['101','1033','123',   '3',  '123',     '123', '999', None,  None]))) 
        self.assertTrue(pd.Series.equals(newcols[7].series, pd.Series([None   ,None, None, '199',   None,    None,   None , None,  None])))         
        
        self.assertTrue(pd.Series.equals(newcols[8].series, pd.Series([False, False, False, False, True,     False,   True, False, False])))   
        

    def testTokenizerCharDecimalOnOnlyEmptyStrings(self):
        newcols = self.adc.Process(self.col4)    
        self.assertEqual(len(newcols), 0)

        
                
        
if __name__ == '__main__':
    unittest.main()
    

