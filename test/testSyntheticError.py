import unittest
import os
import pandas as pd
import numpy as np
import random
from SyntheticError import SyntheticError

import logging
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')


class TestSyntheticError(unittest.TestCase):

    def setUp(self):
        pass
        
    def testInitBadParams1(self):
        # Test bad calls to the constructor.
        with self.assertRaises(TypeError):
            se = SyntheticError(123)
 
    def testInitGood(self):
        # Check we the constructor returns something sane if passed good params.
        se = SyntheticError()
        self.assertIsInstance(se, SyntheticError)

    def testMakeClassificationDefaults(self):
        se = SyntheticError()
        df = se.MakeClassification()
        self.assertIsInstance(df, pd.DataFrame)


    def testMakeClassificationTuned(self):
        se = SyntheticError()
        self.assertIsInstance(se, SyntheticError)        
        df = se.MakeClassification(n_samples=10)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 10) 

        df = se.MakeClassification(n_features=10)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df.columns), 11)       # 10 x's and 1 y.
        
    
    # We should be able to totally predict this dataset.
    @unittest.skipIf(os.environ.get('SKIPSLOW'), 'skipping slow tests')            
    def testVeryPredictable(self):
        rows_list = []
        for i in range(100):
            rows_list.append({ 'a':True,'b':True })
            rows_list.append({ 'a':False,'b':False })

        df = pd.DataFrame(rows_list)
        
        se = SyntheticError()
        howpredictable = se.HowPredictable(df)        
        self.assertEqual(howpredictable, 1.00)        
 
    # And we shouldn't be able to predict this random dataset hardly at all.
    @unittest.skipIf(os.environ.get('SKIPSLOW'), 'skipping slow tests')            
    def testVeryUnPredictable(self):
        rows_list = []
        for i in range(100):
            rows_list.append({ 'a':True,'b':random.randint(0,100) })
            rows_list.append({ 'a':False,'b':random.randint(0,100) })

        df = pd.DataFrame(rows_list)
        
        se = SyntheticError()
        howpredictable = se.HowPredictable(df)   
        # It will be slightly predictable because we group numbers into small ranges.
        self.assertLess(howpredictable, 0.3)        
         
    @unittest.skipIf(os.environ.get('SKIPSLOW'), 'skipping slow tests')            
    def testPredictableMade(self):
        se = SyntheticError()
        df = se.MakeClassification(n_samples=100, n_classes=2, n_features=10, n_redundant=8, n_informative=2)
        howpredictable = se.HowPredictable(df)
        self.assertGreater(howpredictable, 0.1)

    @unittest.skipIf(os.environ.get('SKIPSLOW'), 'skipping slow tests')            
    def testCreateErrors(self):
        se = SyntheticError()
        
        # Try this lots of times because there's some randomness in the choice of error, and we want to make
        # sure there's always exactly as many errors as we ask for.
        for i in range(100):
            df = se.MakeClassification(n_samples=10)        
            df2, boolErrors = se.CreateErrors(df, quantity=5)        
            self.assertEqual(boolErrors.sum().sum(), 5)          
            self.assertFalse(df.equals(df2))

            # We inserted 5 errors, so there should be 5 non-zero differences between the two arrays.
            diff = df.subtract(df2)            
            self.assertEqual(diff.astype(bool).sum().sum(), 5)
if __name__ == '__main__':
    unittest.main()
    

