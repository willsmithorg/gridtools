import unittest
import os
import pandas as pd
import numpy as np
from TrainPredictSelf import TrainPredictSelf

import logging
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')


class TestTrainPredictSelf(unittest.TestCase):

    def setUp(self):
        pass

    @unittest.skipIf(os.environ.get('SKIPSLOW'), 'skipping slow tests')        
    def testBasicClassification(self):
        data = { 'str': ['abc']*10 + ['def']*10,        
                 'num': [0]    *10 + [2]    *10
                }

        #print(data)
        df = pd.DataFrame(data)
        #print(df)
        tps = TrainPredictSelf()
        results_labels, results_proba = tps.Train(df)

        self.assertListEqual(list(results_labels.keys()), ['str', 'num'])
        self.assertListEqual(list(results_labels['str']), ['abc', 'def'])
        self.assertListEqual(list(results_labels['num']), [0,1])        # Will have been label encoded to 0 and 1
   
        # Since this is such a predictable dataframe,
        # we should have the correct predictions and high percentages against them.
        self.assertListEqual(list(results_proba.keys()), ['str', 'num'])
        self.assertTrue(all(results_proba['str'][0:9,0] > 0.8))
        self.assertTrue(all(results_proba['str'][10:19,0] < 0.2))        

        self.assertTrue(all(results_proba['num'][0:9,0] > 0.8))
        self.assertTrue(all(results_proba['num'][10:19,0] < 0.2))  
        
        
    @unittest.skipIf(os.environ.get('SKIPSLOW'), 'skipping slow tests')        
    def testClassificationOnCorrelatedFloats(self):
    
        data = { 'x': np.linspace(0.0, 1.0, 100),
                 'y': np.linspace(1.0, 0.0, 100) 
                }
        df = pd.DataFrame(data)
        tps = TrainPredictSelf()
        
        results_labels, results_proba = tps.Train(df)
        self.assertListEqual(list(results_labels.keys()), ['x', 'y'])

        # The first element should be predicted as the lowest possible.
        self.assertGreater(results_proba['x'][0,0],0.6)
        # The last element should be predicted as the highest possible.
        self.assertGreater(results_proba['x'][99,-1],0.6)     

        # And vice versa for y.
        self.assertGreater(results_proba['y'][0,-1],0.6)
        self.assertGreater(results_proba['y'][99,0],0.6)           


if __name__ == '__main__':
    unittest.main()
    

