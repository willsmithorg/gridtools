import pandas as pd
import numpy as np
from Column import Column
import logging
from ColumnDeriver.Base import ColumnDeriverBase
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')


class ColumnDeriverUpper(ColumnDeriverBase):

    description = "Uppercase of "
    
    # Doesn't make sense to apply this to itself.
    allowrecursive = False

    def IsApplicable(self, column):
        return column.dtype == 'object' and column.Some(column.StrMatches('[a-z]'))
        
    def Apply(self, column):
        newcol = column.series.str.upper()
        return { self.name: newcol }

        
 

    
	