import unittest
import pandas as pd
import numpy as np
from TrainPredictSelf import TrainPredictSelf

import logging
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')


class TestTrainPredictSelf(unittest.TestCase):

    def setUp(self):
        pass
        
    def testBasicClassification(self):
        data = { 'str': ['abc']*10 + ['def']*10,        
                 'num': [0]    *10 + [2]    *10
                }

        #print(data)
        df = pd.DataFrame(data)
        #print(df)
        tps = TrainPredictSelf()
        results = tps.Train(df)
        
        print(results)
        
        
    
        


if __name__ == '__main__':
    unittest.main()
    
