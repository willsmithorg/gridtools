import unittest
import pandas as pd
import numpy as np
from TrainPredict import TrainPredict
from CalcMeanStdPredictions import CalcMeanStdPredictions
from SpotErrors import SpotErrors

import logging
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')

class TestSpotErrors(unittest.TestCase):


    def setUp(self):    
        # Very predictable and totally constant data.
        self.data1 = {'id': np.arange(10),
                 'value': [3.0] * 10}
        self.data1_count = len(self.data1['id'])
        
        # Some predictable data in y.  Mess up 2 items, index 0 and last, and see if we can find them.
        self.data2 = {'x1': [  100,  100,100,100,100,100,100,100,                  200,200,200,200,200,200,200,  200 ],
                      'y':  [-20.0,  1.0,1.0,1.0,1.0,1.0,1.0,1.0,                  6.0,6.0,6.0,6.0,6.0,6.0,6.0,  20.0]}              
        self.data2_count = len(self.data2['y'])
                        

        # Random data.  First is mean 50 std 0.  2nd is mean 200 std 1.
        self.data3_count = 20
        self.data3 = {'col1': [50.0] * self.data3_count,
                      'col2': [200.0] * self.data3_count + np.random.standard_normal(self.data3_count) }

        # Categorical data.
        self.data4_halfcount = 10
        self.data4_count = self.data4_halfcount * 2
        self.data4 = {'categor1': ['A']*self.data4_halfcount+['B']*self.data4_halfcount,
                      'categor2': ['x']*self.data4_halfcount+['y']*self.data4_halfcount}
   
   
    def testNoBadPointsInPredictableData(self):
        tp = TrainPredict()
        ytest = tp.Predict(self.data1, 'value')
        cms = CalcMeanStdPredictions()
        means,stds = cms.Calc(tp, ytest, 'value')
        
        se = SpotErrors()
        boolerrord = se.Spot(tp, means, stds, 'value')
        
        logging.debug(boolerrord)
        # Check boolerrord is the correct shape
        self.assertEqual(boolerrord.shape, (self.data1_count,1))  # 16 rows 1 column
        
        # The predictable array should have no errors.
        for i in range(self.data1_count):        
            with self.subTest(i=i):
                self.assertFalse(boolerrord['value'][i])            
                
                

    def testSpotBadPointsNumeric(self):
        # Check we can spot the deliberate errors introduced into column 2.
        # Check we don't find errors anywhere else.
        tp = TrainPredict()
        ytest = tp.Predict(self.data2, 'y')
        cms = CalcMeanStdPredictions()
        means,stds = cms.Calc(tp, ytest, 'y')
        
        se = SpotErrors()
        boolerrord = se.Spot(tp, means, stds, 'y')

        # Check boolerrord is the correct shape
        self.assertEqual(boolerrord.shape, (self.data2_count,1))  # 16 rows 1 column
        
        
        for i in range(self.data2_count):
            with self.subTest(i=i):
                if i==0 or i==self.data2_count-1:
                    self.assertTrue(boolerrord['y'][i])
                else:
                    self.assertFalse(boolerrord['y'][i])
   

    def testSpotBadPointsNumericSingleRow(self):
        # If we just test on row 0, we shuold get  error.
        tp = TrainPredict()
        ytest = tp.Predict(self.data2, 'y', singlerowid = 0)
        cms = CalcMeanStdPredictions()
        means,stds = cms.Calc(tp, ytest, 'y')        
        se = SpotErrors()
        boolerrord = se.Spot(tp, means, stds, 'y', singlerowid = 0)

        # Check boolerrord is the correct shape
        self.assertEqual(boolerrord.shape, (1,1))  # 1 rows 1 column   
        # And that we get error=True
        self.assertTrue(boolerrord['y'][0])

   
        # If we just test on row 1, we shuold get no error.
        tp = TrainPredict()
        ytest = tp.Predict(self.data2, 'y', singlerowid = 1)
        cms = CalcMeanStdPredictions()
        means,stds = cms.Calc(tp, ytest, 'y')        
        se = SpotErrors()
        boolerrord = se.Spot(tp, means, stds, 'y', singlerowid = 1)

        # Check boolerrord is the correct shape
        self.assertEqual(boolerrord.shape, (1,1))  # 1 rows 1 column   
        # And that we get error=False
        self.assertFalse(boolerrord['y'][0])
        

        # If we just test on end row, we shuold get  error.
        tp = TrainPredict()
        ytest = tp.Predict(self.data2, 'y', singlerowid = self.data2_count-1)
        cms = CalcMeanStdPredictions()
        means,stds = cms.Calc(tp, ytest, 'y')        
        se = SpotErrors()
        boolerrord = se.Spot(tp, means, stds, 'y', singlerowid = self.data2_count-1)

        # Check boolerrord is the correct shape
        self.assertEqual(boolerrord.shape, (1,1))  # 1 rows 1 column   
        # And that we get error=True
        self.assertTrue(boolerrord['y'][0])
        
    def testSpotNoBadPointsInCategoricalData(self):
        # Check we can spot the deliberate errors introduced into column 2.
        # Check we don't find errors anywhere else.
        tp = TrainPredict()
        ytest = tp.Predict(self.data4, 'categor1')
        cms = CalcMeanStdPredictions()
        means,stds = cms.Calc(tp, ytest, 'categor1')
        
        se = SpotErrors()
        boolerrord = se.Spot(tp, means, stds, 'categor1')

        # Check boolerrord is the correct shape
        self.assertEqual(boolerrord.shape, (self.data4_count,2))  # 2 columns because it's been 1-hot encoded
        
        
        for i in range(self.data4_count):
            with self.subTest(i=i):
                if i==0 or i==self.data4_count-1:
                    self.assertFalse(boolerrord['categor1_0'][i])
                    self.assertFalse(boolerrord['categor1_1'][i])

         
if __name__ == '__main__':
    unittest.main()
    
