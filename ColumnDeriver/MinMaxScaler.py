import pandas as pd
import numpy as np
from Column import Column
import logging
from sklearn.preprocessing import MinMaxScaler
from ColumnDeriver.Base import ColumnDeriverBase
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')


class ColumnDeriverMinMaxScaler(ColumnDeriverBase):

    description = "Scaled to the range (0,1) "
    maxdepth = 0
    
    # Doesn't make sense to apply this to itself.    
    allowrecursive = False    

    def __init__(self):
        self.scaler = MinMaxScaler()
                
    def IsApplicable(self, column):
        return column.IsNumeric() 
        
    def Apply(self, column):
        data = column.series
        data = data.to_numpy()
        data = data.reshape(-1,1)
        scaled = self.scaler.fit_transform(data)
        scaled = scaled.reshape(1,-1)[0]        
        series = pd.Series(scaled)
        return { self.name: series }
        
 

    
	