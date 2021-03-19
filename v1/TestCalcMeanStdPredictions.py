import unittest
import pandas as pd
import numpy as np
from TrainPredict import TrainPredict
from CalcMeanStdPredictions import CalcMeanStdPredictions
import logging
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')

class TestCalcMeanStdPredictions(unittest.TestCase):

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

        # Categorical data.
        self.data4_halfcount = 8
        self.data4 = {'categor1': ['A']*self.data4_halfcount+['B']*self.data4_halfcount,
                      'categor2': ['x']*self.data4_halfcount+['y']*self.data4_halfcount}

  
    def testCalcMeanAndDeviation(self):    
        y = [3.0] * 100
        cms = CalcMeanStdPredictions()  
            
        (mn,dev) = cms._CalcMeanAndDeviation(y, 'raw')
        self.assertEqual(mn, 3.0)
        self.assertEqual(dev, 0)
        
        (mn,dev) = cms._CalcMeanAndDeviation(y, 'labelencoded')
        self.assertEqual(mn, 3.0)
        self.assertEqual(dev, 0)
        
        # Normal mean and standard deviation.
        y = [3.0] * 100 + [100]
        (mn,dev) = cms._CalcMeanAndDeviation(y, 'raw')        
        self.assertAlmostEqual(mn, 3.9603960396039604)
        self.assertAlmostEqual(dev, 9.603960396039602)
        
        # If it's a label encoded field, the mean is the mode, and the standard deviation is lower because it's based
        # on the average boolean difference.
        (mn,dev) = cms._CalcMeanAndDeviation(y, 'labelencoded')
        self.assertEqual(mn, 3)
        self.assertAlmostEqual(dev, 0.009900990099009901)
 
 

 
    def testSaneMeanStdPredictionsNumeric(self):
        tp = TrainPredict()
        ytest = tp.Predict(self.data1, 'value')
        cms = CalcMeanStdPredictions()
        means,stds = cms.Calc(tp, ytest, 'value')
        
        
        pd.testing.assert_frame_equal(means[['value']], pd.DataFrame([3.0] * 10, columns=['value']))
        pd.testing.assert_frame_equal(stds[['value']], pd.DataFrame([0.0] * 10, columns=['value']))        

        # Now some standard normal data should have mean and std approximately == 1.0
        tp = TrainPredict()     
        
        # Col1 first.
        ytest = tp.Predict(self.data3, 'col1')
        cms = CalcMeanStdPredictions()
        means,stds = cms.Calc(tp, ytest, 'col1')
        
        pd.testing.assert_frame_equal(means[['col1']], pd.DataFrame([50.0] * self.data3_count, columns=['col1']))
        pd.testing.assert_frame_equal(stds[['col1']], pd.DataFrame([0.0] * self.data3_count, columns=['col1']))      

        # Now col2.
        ytest = tp.Predict(self.data3, 'col2')
        cms = CalcMeanStdPredictions()
        means,stds = cms.Calc(tp, ytest, 'col2')
        
        # Check sane means and standard deviations.  This might randomly fail, if so, check and widen tolerances.
        for i in range(self.data3_count):
            with self.subTest(i=i):
                self.assertTrue(180 <= means['col2'][i] <= 220)
                self.assertTrue(0 <  stds['col2'][i] <= 10)
 
 
                 
    # There are only 2 classes here so we assume (hardcode) that it will be converted into one-hot encoded columns.                
    def testSaneMeanStdPredictionsOneHot(self):
        tp = TrainPredict()
        ytest = tp.Predict(self.data4, 'categor1')
        cms = CalcMeanStdPredictions()
        means,stds = cms.Calc(tp, ytest, 'categor1')
        
        # We need to round because the quantity of data is not enough to give us perfect 1.0 / 0.0 predictions.  The ML
        # is not sure.  But rounding takes us to binary 1.0 and 0.0
        # If this fails in the future, try slightly increasing self.data4_halfcount and hopefully it will succeed with slightly more data.
        # If it doesn't increase with self.data4_halfcount set high, we have a genuine problem.
        pd.testing.assert_frame_equal(round(means[['categor1_0']]), pd.DataFrame([1.0]*self.data4_halfcount+[0.0]*self.data4_halfcount, columns=['categor1_0']))
        pd.testing.assert_frame_equal(round(means[['categor1_1']]), pd.DataFrame([0.0]*self.data4_halfcount+[1.0]*self.data4_halfcount, columns=['categor1_1']))




         
if __name__ == '__main__':
    unittest.main()
