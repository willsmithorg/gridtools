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
        f = MakeFrameNumeric()

        self.assertIsInstance(f, MakeFrameNumeric)

    
    def testSetCardinalityForOneHot(self):
        # Check we can override the maximum cardinality for one-hot encode.
        f = MakeFrameNumeric()
        
        self.assertNotEqual(f.maximum_cardinality_for_one_hot_encode, 123)
        f.maximum_cardinality_for_one_hot_encode = 123
        self.assertEqual(f.maximum_cardinality_for_one_hot_encode, 123)
    

    def testConvertAllNumeric(self):
        # If we pass in an all-numeric dataframe, it should not be converted at all.
        f = MakeFrameNumeric()
         
        converted = f.Convert(self.validDataFrame)
         
        self.assertIsInstance(converted, pd.DataFrame)
        self.assertTrue(converted.equals(self.validDataFrame))
        
 
    def testConvertSomeString(self):
        # If we start with a string column with low cardinality, it should get converted to a one-hot column.
        data = {'id':   [1,2,3],
                'value':['A', 'B', 'C']}
        df = pd.DataFrame(data)
        f = MakeFrameNumeric()

        f.maximum_cardinality_for_one_hot_encode = 3         
        converted = f.Convert(df)
         
        self.assertIsInstance(converted, pd.DataFrame)
        
        self.assertEqual(converted.shape, (3,4))
        
        self.assertIsInstance(converted.iloc[0,0], (np.int32, np.int64, np.float64))        # id
        self.assertIsInstance(converted.iloc[0,1], (np.int32, np.int64, np.float64))        # 2nd column - onehot
        self.assertIsInstance(converted.iloc[0,2], (np.int32, np.int64, np.float64))        # 3nd column - onehot
        self.assertIsInstance(converted.iloc[0,3], (np.int32, np.int64, np.float64))        # 4nd column - onehot
        
        # If we set the maximum cardinality lower than the column's cardinality, we should instead get a single converted column.        
        f.maximum_cardinality_for_one_hot_encode = 2
        converted  = f.Convert(df)         
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
        f = MakeFrameNumeric()        
        f.maximum_cardinality_for_one_hot_encode = 3         
        converted = f.Convert(df)
        self.assertIsInstance(converted, pd.DataFrame)
        self.assertEqual(converted.shape, (10,1+20+(2*20)))  # 1 for the id, 20 for the high-cardinality column, and 2 for each of 20 one-hots, for the low cardinality columns
         

    def testColMapFeatureMap(self):
        data = { 'country': ['Germany','Germany','Germany','Germany','Germany',
                             'US', 'US', 'US', 'US', 'US',
                             'UK', 'UK', 'UK', 'UK', 'UK'],
                 'currency':['EUR','EUR','EUR','EUR','MXN',
                             'USD','USD','USD','USD','GBP',
                             'GBP','GBP','GBP','GBP','GBP'],
                 'manyvalues':['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']}
            
        frame = pd.DataFrame(data)
        mfn = MakeFrameNumeric()
        mfn.maximum_cardinality_for_one_hot_encode = 10   # Hardcode so we don't get unexpected behaviour below if we change the default in the class.

        converted = mfn.Convert(frame)
        

        # The colmapd2s is a mapping from destination to source column.
        self.assertEqual(mfn.colmapd2s, 
            {'country_0': 'country', 'country_1': 'country', 'country_2': 'country', 
            'currency_0': 'currency', 'currency_1': 'currency', 'currency_2': 'currency', 'currency_3': 'currency', 
            'manyvalues': 'manyvalues'})
            
        # The colmaps2d is a mapping from source to destinationcolumn.
        self.assertEqual(mfn.colmaps2d, 
            {'country': ['country_0', 'country_1', 'country_2' ], 
             'currency': ['currency_0', 'currency_1', 'currency_2', 'currency_3' ], 
             'manyvalues': ['manyvalues']})            
            
        # The featuremap is a mapping from converted value to source token, per destination column.
        self.assertEqual(mfn.featuremapd, 
            {'country_0': 'Germany', 'country_1': 'UK', 'country_2': 'US', 
             'currency_0': 'EUR', 'currency_1': 'GBP', 'currency_2': 'MXN', 'currency_3': 'USD', 
             'manyvalues': {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 
                            9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O'}})

        self.assertCountEqual(mfn.featuremaps['country'], {'Germany':0, 'UK':1, 'US':2})
        self.assertCountEqual(mfn.featuremaps['currency'], {'EUR':0, 'GBP':1, 'MXN':2, 'USD':3});
        
        self.assertEqual(mfn.featuremaps['manyvalues'],  {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8, 
                            'J':9, 'K':10, 'L':11, 'M':12, 'N':13, 'O':14})
                            
        self.assertEqual(mfn.coltyped, 
            {'country_0': 'onehot', 'country_1': 'onehot', 'country_2': 'onehot', 
             'currency_0': 'onehot', 'currency_1': 'onehot', 'currency_2': 'onehot', 'currency_3': 'onehot', 
             'manyvalues': 'labelencoded'})
        
    
if __name__ == '__main__':
    unittest.main()
    