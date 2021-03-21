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
    allowrecursive = False    

    def __init__(self):
        print('abs::init')
                
    def IsApplicable(self, column):
        return column.dtype == 'int64' and any(column.series > 0) and any(column.series < 0)
        
    def Apply(self, column):
        newcol = Column(column.series.map(abs))
        return newcol

        
 

    
	