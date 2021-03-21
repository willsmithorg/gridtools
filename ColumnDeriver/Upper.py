import pandas as pd
import numpy as np
from Column import Column
import logging
from ColumnDeriver.Base import ColumnDeriverBase
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')


class ColumnDeriverUpper(ColumnDeriverBase):

    name = "uppercase"
    description = "Uppercase of "
    allowrecursive = False

    def __init__(self):
        print('upper::init')        
        
    def IsApplicable(self, column):
        return column.dtype == 'object'
        
    def Apply(self, column):
        newcol = Column(column.series.str.upper())
        return newcol

        
 

    
	