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
                      'y':  [-20.0,  1.0,1.0,1.0,1.0,1.0,1.0,1.0,                  6.0,6.0,6.0,6.0,6.0,6.0,6.0,  20.0]}              
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
        
        print(means)
        pd.testing.assert_frame_equal(means[['value']], pd.DataFrame([3.0] * 10, columns=['value']))
        pd.testing.assert_frame_equal(stds[['value']], pd.DataFrame([0.0] * 10, columns=['value']))        

        # Now some standard normal data should have mean and std approximately == 1.0
        tp = TrainPredict()     
        means,stds = tp.Predict(self.data3)
  
        pd.testing.assert_frame_equal(means[['col1']], pd.DataFrame([50.0] * self.data3_count, columns=['col1']))
        pd.testing.assert_frame_equal(stds[['col1']], pd.DataFrame([0.0] * self.data3_count, columns=['col1']))      

        # Check sane means and standard deviations.  This might randomly fail, if so, check and widen tolerances.
        for i in range(self.data3_count):
            with self.subTest(i=i):
                self.assertTrue(180 <= means['col2'][i] <= 220)
                self.assertTrue(0 <  stds['col2'][i] <= 10)
        
    def testSpotBadPoints(self):
        # Check we can spot the deliberate errors introduced into column 2.
        # Check we don't find errors anywhere else.
        tp = TrainPredict() 
        boolErrors = tp.SpotErrors(self.data2)
        #print(boolErrors)
        for i in range(self.data2_count):
            with self.subTest(i=i):
                if i==0 or i==self.data2_count-1:
                    self.assertTrue(boolErrors['y'][i])
                else:
                    self.assertFalse(boolErrors['y'][i])
     
     
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
        
        # Now repeat on numpy arrays.
        y = np.array([[1.0, 2.0, 3.0], 
                     [3.0, 4.0, 5.0]])
        (mn,dev) = tp.CalcMeanAndDeviation(y, 'raw')
        self.assertTrue(np.allclose(mn, [2.0, 4.0]))
        self.assertTrue(np.allclose(dev, [0.81649658, 0.81649658]))
        
        (mn,dev) = tp.CalcMeanAndDeviation(y, 'labelencoded')
        print(mn)
        print(dev)
        self.assertTrue(np.allclose(mn, [1.0, 3.0]))         # mode returns the first value if there is no mode
        self.assertTrue(np.allclose(dev, [0.66666667, 0.66666667]))
        
        
        
if __name__ == '__main__':
    unittest.main()
    