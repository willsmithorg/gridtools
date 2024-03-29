import pandas as pd
import numpy as np
from Column import Column
import logging
from sklearn.preprocessing import RobustScaler
from ColumnDeriver.Base import ColumnDeriverBase
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')


class ColumnDeriverRobustScaler(ColumnDeriverBase):

    description = "Scaled to the range (0,1) with outliers handled "
    maxdepth = 0
    
    # Doesn't make sense to apply this to itself.    
    allowrecursive = False 
    
    # Any scaling by definition is likley to be 1:1 but we still allow it.
    allowOneToOne = True
    
    def __init__(self):
        self.scaler = RobustScaler(unit_variance = True)
                
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

        
 

    
	