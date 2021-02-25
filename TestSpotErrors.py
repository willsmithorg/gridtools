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
        self.data2 = {'x1': [  100,  100,100,100,100,100,100,100,100,100,                  200,200,200,200,200,200,200,200,200,  200 ],
                      'y':  [-20.0,  1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,                  6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,  20.0]}              
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
   
   
        # Mixture of categorical and non-categorical data.
        self.data5 = { 'country': ['Germany','Germany','Germany','Germany','Germany', 'Germany',
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
                         'linear': [ x for x in range(21) ]}
        self.data5_count = len(self.data5['country'])                         
         
   
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
        
    # This tests boolean error forecasts at the destination column level.
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
        
        # Check all points are false error.
        for i in range(self.data4_count):
            with self.subTest(i=i):
                if i==0 or i==self.data4_count-1:
                    self.assertFalse(boolerrord['categor1_0'][i])
                    self.assertFalse(boolerrord['categor1_1'][i])

    # This tests boolean error forecasts and predictions at the source column level.         
    def testGetErrorsAndPredictionsTest4Dataset(self):
    
        # Test4 dataset.

        # categor1 column.
        tp = TrainPredict()
        ytest = tp.Predict(self.data4, 'categor1')
        means,stds = CalcMeanStdPredictions().Calc(tp, ytest, 'categor1')        
        se = SpotErrors()
        boolerrord = se.Spot(tp, means, stds, 'categor1')   
        boolerrors, predictions = se.GetErrorsAndPredictions('categor1')
 
        # Check correct shape.
        self.assertEqual(boolerrors.shape, (self.data4_count,1))
        self.assertEqual(predictions.shape, (self.data4_count,1))        

        # Check all errors are false and all predictions are empty lists.
        for i in range(self.data4_count):
            with self.subTest(i=i):
                if i==0 or i==self.data4_count-1:
                    self.assertFalse(boolerrors['categor1'][i])
                    self.assertEqual(predictions['categor1'][i], [])

        # categor2 column.
        tp = TrainPredict()
        ytest = tp.Predict(self.data4, 'categor2')
        means,stds = CalcMeanStdPredictions().Calc(tp, ytest, 'categor2')        
        se = SpotErrors()
        boolerrord = se.Spot(tp, means, stds, 'categor2')   
        boolerrors, predictions = se.GetErrorsAndPredictions('categor2')
 
        # Check correct shape.
        self.assertEqual(boolerrors.shape, (self.data4_count,1))
        self.assertEqual(predictions.shape, (self.data4_count,1))        

        # Check all errors are false and all predictions are empty lists.
        for i in range(self.data4_count):
            with self.subTest(i=i):
                if i==0 or i==self.data4_count-1:
                    self.assertFalse(boolerrors['categor2'][i])
                    self.assertEqual(predictions['categor2'][i], [])
        
    def testGetErrorsAndPredictionsTest2Dataset(self):

        # Test2 dataset.
        # Check we can spot the deliberate errors introduced into column 2.
        # Check we don't find errors anywhere else.
        tp = TrainPredict()
        ytest = tp.Predict(self.data2, 'y')
        cms = CalcMeanStdPredictions()
        means,stds = cms.Calc(tp, ytest, 'y')        
        se = SpotErrors()
        boolerrord = se.Spot(tp, means, stds, 'y')
        boolerrors, predictions = se.GetErrorsAndPredictions('y')

        # Check correct shape.
        self.assertEqual(boolerrors.shape, (self.data2_count,1))
        self.assertEqual(predictions.shape, (self.data2_count,1))        
        
        logging.debug(boolerrors)
        logging.debug(predictions)
        
        for i in range(self.data2_count):
            with self.subTest(i=i):
                if i==0:
                    self.assertTrue(boolerrors['y'][i])
                    self.assertEqual(len(predictions['y'][i]), 1)
                    self.assertTrue(-3 <= predictions['y'][i][0] <= 3)  # strictly, should be 1.0 ish
                elif i==self.data2_count-1:
                    self.assertTrue(boolerrors['y'][i])
                    self.assertEqual(len(predictions['y'][i]), 1)                    
                    self.assertTrue(4 <= predictions['y'][i][0] <= 8)  # strictly, should be 6.0 ish
                else:
                    self.assertFalse(boolerrors['y'][i])  
                    self.assertEqual(predictions['y'][i], [])

    def testGetErrorsAndPredictionsTest3Dataset(self):

        # Test3 dataset.

        # col1 is unpredictable because col2 is random.  Hence, no errors.
        tp = TrainPredict()
        ytest = tp.Predict(self.data3, 'col1')
        means,stds = CalcMeanStdPredictions().Calc(tp, ytest, 'col1')        
        se = SpotErrors()
        boolerrord = se.Spot(tp, means, stds, 'col1')   
        boolerrors, predictions = se.GetErrorsAndPredictions('col1')
 
        # Check correct shape.
        self.assertEqual(boolerrors.shape, (self.data3_count,1))
        self.assertEqual(predictions.shape, (self.data3_count,1))        

        # Check all errors are false and all predictions are empty lists.
        for i in range(self.data3_count):
            with self.subTest(i=i):
                if i==0 or i==self.data3_count-1:
                    self.assertFalse(boolerrors['col1'][i])
                    self.assertEqual(predictions['col1'][i], [])
                    
        # col1 is random, and unpredictable because col1 is constant.  Hence, no errors.
        tp = TrainPredict()
        ytest = tp.Predict(self.data3, 'col2')
        means,stds = CalcMeanStdPredictions().Calc(tp, ytest, 'col2')        
        se = SpotErrors()
        boolerrord = se.Spot(tp, means, stds, 'col2')   
        boolerrors, predictions = se.GetErrorsAndPredictions('col2')
 
        # Check correct shape.
        self.assertEqual(boolerrors.shape, (self.data3_count,1))
        self.assertEqual(predictions.shape, (self.data3_count,1))        

        logging.debug(boolerrors['col2'])
        logging.debug(self.data3)
        logging.debug(means)
        logging.debug(stds)
        # Check all errors are false and all predictions are empty lists.
        for i in range(self.data3_count):
            with self.subTest(i=i):
                if i==0 or i==self.data3_count-1:
                    self.assertFalse(boolerrors['col2'][i])
                    self.assertEqual(predictions['col2'][i], [])
 
    def testGetErrorsAndPredictionsTest5Dataset(self):

        # Test5 dataset.
        # 'manyvalues' has a deliberate error in row 20.  It should be 'Z' not 'Y'.
        tp = TrainPredict()
        ytest = tp.Predict(self.data5, 'manyvalues')
        means,stds = CalcMeanStdPredictions().Calc(tp, ytest, 'manyvalues')        
        se = SpotErrors()
        boolerrord = se.Spot(tp, means, stds, 'manyvalues')   
        boolerrors, predictions = se.GetErrorsAndPredictions('manyvalues') 
        
        for i in range(self.data5_count):
            with self.subTest(i=i):
                if i==20:
                    self.assertTrue(boolerrors['manyvalues'][i])
                    self.assertEqual(len(predictions['manyvalues'][i]), 1)
                    self.assertEqual(predictions['manyvalues'][i][0], 'Z') # is Y, should be Z.               
                else:
                    self.assertFalse(boolerrors['manyvalues'][i])  
                    self.assertEqual(predictions['manyvalues'][i], [])
                    
if __name__ == '__main__':
    unittest.main()
    
