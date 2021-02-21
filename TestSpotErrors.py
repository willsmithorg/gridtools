import unittest
import pandas as pd
import numpy as np
from TrainPredict import TrainPredict
from SpotErrors import SpotErrors

class TestSpotErrors(unittest.TestCase):


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
   
   

    def testSpotBadPointsNumeric(self):
        # Check we can spot the deliberate errors introduced into column 2.
        # Check we don't find errors anywhere else.
        tp = TrainPredict() 
        boolErrors = tp.SpotErrors(self.data2)
        for i in range(self.data2_count):
            with self.subTest(i=i):
                if i==0 or i==self.data2_count-1:
                    self.assertTrue(boolErrors['y'][i])
                else:
                    self.assertFalse(boolErrors['y'][i])
   


