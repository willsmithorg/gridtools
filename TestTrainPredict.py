import unittest
import pandas as pd
import numpy as np
from TrainPredict import TrainPredict


class TestTrainPredict(unittest.TestCase):

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
        self.data4_halfcount = 10
        self.data4 = {'categor1': ['A']*self.data4_halfcount+['B']*self.data4_halfcount,
                      'categor2': ['x']*self.data4_halfcount+['y']*self.data4_halfcount}
                      
        
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

    
    def testPredictWrongColErrors(self):
        tp = TrainPredict()
        with self.assertRaises(ValueError):
            ytest = tp.Predict(self.data1, "badcolumnname")

    # Prediction should be n_categories(e.g. onehot) * nmodels * nrows.
    def testPredictionsShape(self):
        tp = TrainPredict()
        tp.models_for_confidence = 8
        ytest = tp.Predict(self.data1, 'id')
        self.assertEqual(ytest.shape, (1, 8, 10))          # 8 copies of the model, trained on 10 rows.
        
        ytest = tp.Predict(self.data1, 'value')
        self.assertEqual(ytest.shape, (1, 8, 10))          # 8 copies of the model, trained on 10 rows.
         
        # Categorical data should have been 1-hot encoded.
        ytest = tp.Predict(self.data4, 'categor1') 
        self.assertEqual(ytest.shape, (2, 8, 2 * self.data4_halfcount))       

        # Train on only 1 row - the last row.
        ytest = tp.Predict(self.data4, 'categor1', singlerowid=9) 
        self.assertEqual(ytest.shape, (2, 8, 1))       
        
        # Try to train on row 1-million.
        with self.assertRaises(ValueError):
            ytest = tp.Predict(self.data4, 'categor1', singlerowid=1000000) 
                
        
        
if __name__ == '__main__':
    unittest.main()
    