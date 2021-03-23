import pandas as pd
import numpy as np
from Column import Column
import logging
from ColumnDeriver.Base import ColumnDeriverBase
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')


class ColumnDeriverAbs(ColumnDeriverBase):

    name = "absolute"
    description = "The absolute value of "
    maxdepth = 1
    
    # Doesn't make sense to apply this to itself.    
    allowrecursive = False    
        
    def IsApplicable(self, column):
        return self.IsNumeric(column) and any(column.series > 0) and any(column.series < 0)
        
    def Apply(self, column):
        newcol = column.series.map(abs)
        return { self.name: newcol }

        
 

    
	