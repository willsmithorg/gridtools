import unittest
import pandas as pd
from TestMakeFrameNumeric import TestMakeFrameNumeric


class TestMakeFrameNumeric(unittest.TestCase):

    def testInitNotDataFrame(self):
        with self.assertRaises(TypeError):
            f = MakeFrameNumeric(123)
            
        with self.assertRaises(TypeError):
            data = {'id':   [1],
                    'value':[3]}
                    
            df = pd.DataFrame(data)
            f = MakeFrameNumeric(df)
            
