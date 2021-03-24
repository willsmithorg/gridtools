import pandas as pd
import numpy as np
import logging


logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')


class ColumnDeriverBase:


    name = ""
    description = ""
    # For sanity, unless we specify maxdepth in the derived class.  We don't want infinite recursion!
    maxdepth = 2
    
    # Can this deriver be applied to a column that was created from this deriver somewhere up the hierarchy?
    # TODO implement.
    allowrecursive = False

    def __init__(self):
        pass
        
    def IsApplicable(self, column):
        return True
        
    def __str__(self):
        n =             'Name : ' + name + '\n'        
        subclasses =    'Subclasses : ' + ','.join([cls.__name__ for cls in ColumnDeriverBase.__subclasses__()]) + '\n'
        return n + subclasses
        
    def IsNumeric(self, column):
        return column.dtype == 'int64' or column.dtype == 'float64'
        
    def GetDerivers(self):
        return ColumnDeriverBase.__subclasses__()
        
    def StrContains(self, column, substring):
        return column.series.str.contains(substring, regex=False)
        
    def StrMatches(self, column, regex):
        return column.series.str.contains(regex, regex=True)

    def All(self, column, boolseries, threshold=1.0):
        if boolseries.sum() >= threshold * column.size:
            return True
        else:
            return False        
        
    def Most(self, column, boolseries, threshold=0.8):
        if boolseries.sum() >= threshold * column.size:
            return True
        else:
            return False
            
    def Some(self, column, boolseries, threshold=0.5):
        if boolseries.sum() >= threshold * column.size:
            return True
        else:
            return False            
        
    def AFew(self, column, boolseries, threshold=0.2):
        if boolseries.sum() >= threshold * column.size:
            return True
        else:
            return False   

    def NotAny(self, column, boolseries, threshold=0.0):
        if boolseries.sum() <= threshold * column.size:
            return True
        else:
            return False             