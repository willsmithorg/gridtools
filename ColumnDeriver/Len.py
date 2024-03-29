import pandas as pd
import numpy as np
from Column import Column
import logging
from ColumnDeriver.Base import ColumnDeriverBase
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')


class ColumnDeriverLen(ColumnDeriverBase):

    description = "Length of "
    maxdepth = 2

    allowOneToOne = False
    
    def IsApplicable(self, col):  
        # print('considering len() on', col.name, 'its type is ', col.dtype)        
        return col.dtype == 'object' and not col.IsNumeric()
        
    def Apply(self, col):
        # Replace nones with '' so they don't error, and the length can be 0.
        newcol = col.series.fillna('').map(len)      
        return { self.name: newcol }

        
 

    
	