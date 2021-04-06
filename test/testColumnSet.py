import unittest
import pandas as pd
import numpy as np
import copy
from Column import Column
from ColumnSet import ColumnSet
from AddDerivedColumns import AddDerivedColumns

import logging
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')


class TestColumnSet(unittest.TestCase):


    def setUp(self):
    
        data1 = { 
             'country': ['Germany','Germany','Germany','Germany','Germany', 'Germany',
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
             'linear': [ x for x in range(21) ],
             'city': ['Frankfurt'] * 6 + ['LA'] * 5 + ['London'] * 5+  ['Luxembourg'] * 5}
            
        self.data1 = pd.DataFrame(data1) 


    def testInitBad(self):
        # Test bad calls to the constructor.
        with self.assertRaises(TypeError):
            f = ColumnSet(123)
    
        with self.assertRaises(TypeError):
            f = ColumnSet()    
            
            
    def testInitGood(self):
        # Check we the constructor returns something sane if passed good params.
        f = ColumnSet(self.data1)
        self.assertIsInstance(f, ColumnSet)    

    def testInitGood(self):
        # Check we the constructor returns something sane if passed good params.
        f = ColumnSet(self.data1)
        self.assertIsInstance(f, ColumnSet)  
    
    def testSize(self):
        f = ColumnSet(pd.DataFrame([]))        
        self.assertEqual(f.size, 0)

        f = ColumnSet(self.data1)
        self.assertEqual(f.size, 21)
        
    def testAddDerived(self):
        cs1 = ColumnSet(self.data1)
        
        self.assertGreater(len(cs1.inputcolumns),0)
        self.assertEqual(len(cs1.derivedcolumns),0)
        
        adc = AddDerivedColumns()
        adc.RegisterDefaultDerivers()
        cs1.AddDerived(adc)
        
        self.assertGreater(len(cs1.inputcolumns),0)
        self.assertGreater(len(cs1.derivedcolumns),0)               

    def testRemove(self):
        # print(len(self.data1.columns))
        cs1 = ColumnSet(self.data1)
        adc = AddDerivedColumns()
        adc.RegisterDefaultDerivers()
        cs1.AddDerived(adc)
        
        inputbefore   = len(cs1.inputcolumns)
        derivedbefore = len(cs1.derivedcolumns)
        
        #print(', '.join([x.name for x in cs1.inputcolumns]))
        # There are currently 6 input columns.
        self.assertEqual(inputbefore, 6)
 
        # There are lots of derived columns.
        self.assertGreater(derivedbefore, 6) 
        
        cs2 = copy.copy(cs1)
        
        # Now remove a source column.  The number of derived columns should also drop.
        
        cs1.Remove('country')
        inputafter   = len(cs1.inputcolumns)
        derivedafter = len(cs1.derivedcolumns)   

        # One few input columns
        self.assertEqual(inputbefore - inputafter, 1)
        # Several fewer derived columns
        self.assertGreater(derivedbefore - derivedafter, 1)

        # And the copy of the columnset should not have been altered.
        inputbefore2   = len(cs2.inputcolumns)
        derivedbefore2 = len(cs2.derivedcolumns)    
        self.assertEqual(inputbefore2, inputbefore) 
        # There are lots of derived columns.
        self.assertEqual(derivedbefore2, derivedbefore)        


    def testGetInputColumnNames(self):
        cs1 = ColumnSet(self.data1)    
        self.assertEqual(cs1.GetInputColumnNames(), ['country', 'currency', 'manyvalues', 'rand', 'linear', 'city' ])
        
 
    def testGetInputColumn(self):
        cs1 = ColumnSet(self.data1) 
        col = cs1.GetInputColumn('country')
        self.assertTrue(pd.Series.equals(col.series, pd.Series(self.data1['country'])))

        cs1 = ColumnSet(self.data1) 
        col = cs1.GetInputColumn('this doesnt exist')
        self.assertEqual(col, None)
        
        
if __name__ == '__main__':
    unittest.main()
    