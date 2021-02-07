import unittest
import pandas as pd
import numpy as np
from MakeFrameNumeric import MakeFrameNumeric


class TestMakeFrameNumeric(unittest.TestCase):

    def setUp(self):
        data = {'id':   [1],
                'value':[3]}                
        self.validDataFrame = pd.DataFrame(data)
        
        
    
    def testInitBadParams1(self):
        # Test bad calls to the constructor.
        with self.assertRaises(TypeError):
            f = MakeFrameNumeric(123)

    def testInitBadParams2(self):            
        # Test bad calls to the constructor.
        with self.assertRaises(TypeError):
            f = MakeFrameNumeric([])
 
    def testInitGood(self):
        # Check we the constructor returns something sane if passed good params.
        f = MakeFrameNumeric(self.validDataFrame)

        self.assertIsInstance(f, MakeFrameNumeric)

    
    def testSetCardinalityForOneHot(self):
        # Check we can override the maximum cardinality for one-hot encode.
        f = MakeFrameNumeric(self.validDataFrame)
        
        self.assertNotEqual(f.maximum_cardinality_for_one_hot_encode, 123)
        f.maximum_cardinality_for_one_hot_encode = 123
        self.assertEqual(f.maximum_cardinality_for_one_hot_encode, 123)
    

    def testConvertAllNumeric(self):
        # If we pass in an all-numeric dataframe, it should not be converted at all.
        f = MakeFrameNumeric(self.validDataFrame)
         
        converted = f.ConvertForXGBoost()
         
        self.assertIsInstance(converted, pd.DataFrame)
        self.assertTrue(converted.equals(self.validDataFrame))
        
 
    def testConvertSomeString(self):
        # If we start with a string column with low cardinality, it should get converted to a one-hot column.
        data = {'id':   [1,2,3],
                'value':['A', 'B', 'C']}
        df = pd.DataFrame(data)
        f = MakeFrameNumeric(df)

        f.maximum_cardinality_for_one_hot_encode = 3         
        converted = f.ConvertForXGBoost()
         
        self.assertIsInstance(converted, pd.DataFrame)
        
        self.assertEqual(converted.shape, (3,4))
        
        self.assertIsInstance(converted.iloc[0,0], (np.int64, np.float64))        # id
        self.assertIsInstance(converted.iloc[0,1], (np.int64, np.float64))        # 2nd column - onehot
        self.assertIsInstance(converted.iloc[0,2], (np.int64, np.float64))        # 3nd column - onehot
        self.assertIsInstance(converted.iloc[0,3], (np.int64, np.float64))        # 4nd column - onehot
        
        # If we set the maximum cardinality lower than the column's cardinality, we should instead get a single converted column.        
        f.maximum_cardinality_for_one_hot_encode = 2
        converted = f.ConvertForXGBoost()         
        self.assertIsInstance(converted, pd.DataFrame)
        self.assertEqual(converted.shape, (3,2))
        
    def testConvertLargeTable(self):
       
        dict = {}
        dict['id']= [ 'A','B','C','D','E','F','G','H','I','J' ]
        for col in range(20):
            dict['label'+str(col)] = [ 'A','B','C','D','E','F','G','H','I','J']
        
        for col in range(20):
            dict['onehot'+str(col)] = [ 'A','A','A','A','A','B','B','B','B','B']

        df = pd.DataFrame(dict)
        f = MakeFrameNumeric(df)        
        f.maximum_cardinality_for_one_hot_encode = 3         
        converted = f.ConvertForXGBoost()
        self.assertIsInstance(converted, pd.DataFrame)
        self.assertEqual(converted.shape, (10,1+20+(2*20)))  # 1 for the id, 20 for the high-cardinality column, and 2 for each of 20 one-hots, for the low cardinality columns
         

if __name__ == '__main__':
    unittest.main()
    