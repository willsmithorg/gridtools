import pandas as pd
import numpy as np
from Column import Column
import logging
from ColumnDeriver.Base import ColumnDeriverBase
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')


class ColumnDeriverLen(ColumnDeriverBase):

    name = "length"
    description = "Length of "
    maxdepth = 2

    def IsApplicable(self, col):
        print('considering len() on', col.name, 'its type is ', col.dtype)
        
        return col.dtype == 'object' and ~ self.IsNumeric(col)
        
    def Apply(self, col):
        newcol = col.series.map(len)      
        return { self.name: newcol }

        
 

    
	