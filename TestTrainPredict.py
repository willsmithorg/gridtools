import unittest
import pandas as pd
import numpy as np
from TrainPredict import TrainPredict


class TestMakeFrameNumeric(unittest.TestCase):

    def setUp(self):    
        # Very predictable and totally constant data.
        self.data1 = {'id': np.arange(10),
                 'value': [3.0] * 10}
        
        # Some predictable data in y.  Mess up 2 items, index 0 and last, and see if we can find them.
        self.data2 = {'x1': [  100,  100,100,100,100,100,100,100,                  200,200,200,200,200,200,200,  200 ],
                      'y':  [-20.0,  1.0,1.0,1.0,1.0,1.0,1.0,1.0,                  6.0,5.0,6.0,5.0,6.0,5.0,6.0,  20.0]}              
        self.data2_count = len(self.data2['y'])
                        

        # Random data.  First is mean 50 std 0.  2nd is mean 200 std 1.
        self.data3_count = 20
        self.data3 = {'col1': [50.0] * self.data3_count,
                      'col2': [200.0] * self.data3_count + np.random.standard_normal(self.data3_count) }

        
    def testInitBadParams1(self):
        # Test bad calls to the constructor.
        with self.assertRaises(TypeError):
            tp = TrainPredict(123)

    def testInitBadParams2(self):            
        # Test bad calls to the constructor.
        with self.assertRaises(TypeError):
            tp = TrainPredict([])
 
    def testInitGood(self):
        # Check we the constructor returns something sane if passed good params.
        tp = TrainPredict()
        self.assertIsInstance(tp, TrainPredict)

    def testSaneMeanStdPredictions(self):
        tp = TrainPredict()         
        means,stds = tp.Predict(self.data1)
        
        self.assertTrue(all(means[:,1] == [3.0] * 10))
        self.assertTrue(all(stds[:,1] == [0.0] * 10))

        # Now some standard normal data should have mean and std approximately == 1.0
        tp = TrainPredict()     
        means,stds = tp.Predict(self.data3)
  
        self.assertTrue(all(means[:,0] == [50.0] * self.data3_count))
        self.assertTrue(all(stds[:,0] == [0.0] * self.data3_count))

        # Check sane means and standard deviations.  This might randomly fail, if so, check and widen tolerances.
        for i in range(self.data3_count):
            with self.subTest(i=i):
                self.assertTrue(180 <= means[i,1] <= 220)
                self.assertTrue(0.00001 <  stds[i,1] <= 10)
        

    def testSpotBadPoints(self):
        # Check we can spot the deliberate errors introduced into column 2.
        # Check we don't find errors anywhere else.
        tp = TrainPredict() 
        boolErrors = tp.SpotErrors(self.data2)
        #print(boolErrors)
        for i in range(self.data2_count):
            with self.subTest(i=i):
                if i==0 or i==self.data2_count-1:
                    self.assertTrue(boolErrors[i][1])
                else:
                    self.assertFalse(boolErrors[i][1])
     
     
    def testCalcMeanAndDeviation(self):    
        y = [3.0] * 100
        tp = TrainPredict()              
        (mn,dev) = tp.CalcMeanAndDeviation(y, 'raw')
        self.assertEqual(mn, 3.0)
        self.assertEqual(dev, 0)
        
        (mn,dev) = tp.CalcMeanAndDeviation(y, 'labelencoded')
        self.assertEqual(mn, 3.0)
        self.assertEqual(dev, 0)
        
        # Normal mean and starndard deviation.
        y = [3.0] * 100 + [100]
        (mn,dev) = tp.CalcMeanAndDeviation(y, 'raw')        
        self.assertAlmostEqual(mn, 3.9603960396039604)
        self.assertAlmostEqual(dev, 9.603960396039602)
        
        # If it's a label encoded field, the mean is the mode, and the standard deviation is lower because its' based
        # on the average boolean difference.
        (mn,dev) = tp.CalcMeanAndDeviation(y, 'labelencoded')
        self.assertEqual(mn, 3)
        self.assertAlmostEqual(dev, 0.009900990099009901)
        
if __name__ == '__main__':
    unittest.main()
    