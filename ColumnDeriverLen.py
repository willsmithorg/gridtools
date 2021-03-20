import pandas as pd
import numpy as np
import logging
import Column as Column
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')


class ColumnDeriverLen(ColumnDeriverBase):

    name = "length"
    description = "Length of "
    maxdepth = 0
    
    def Apply(column):
        col = Column(col.series.map(len))
        return col
        
    def IsApplicable(column)
        return column.dtype == 'object'
        
    
    
	