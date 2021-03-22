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
        return col.dtype == 'object'
        
    def Apply(self, col):
        newcol = Column(col.series.map(len))       
        return { self.name: newcol }

        
 

    
	