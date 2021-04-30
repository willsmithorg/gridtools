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
        results = tps.Train(df)
        
        # print(results)

    @unittest.skipIf(os.environ.get('SKIPSLOW'), 'skipping slow tests')        
    def testClassificationOnCorrelatedFloats(self):
    
        data = { 'x': np.linspace(0.0, 1.0, 100),
                 'y': np.linspace(1.0, 0.0, 100) 
                }
        df = pd.DataFrame(data)
        tps = TrainPredictSelf()
        results = tps.Train(df)
        
        #print(results)
        


if __name__ == '__main__':
    unittest.main()
    

