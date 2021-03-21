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

    
    def IsApplicable(self, column):
        return True
        
    def __str__(self):   
        return ','.join([cls.__name__ for cls in ColumnDeriverBase.__subclasses__()])
        
    def GetDerivers(self):
        return ColumnDeriverBase.__subclasses__()
        