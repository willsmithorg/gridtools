import pandas as pd
import numpy as np
import logging


logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')


class ColumnDeriverBase:
    
    description = ""
    # For sanity, unless we specify maxdepth in the derived class.  We don't want infinite recursion!
    maxdepth = 2
    
    # Can this deriver be applied to a column that was created from this deriver somewhere up the hierarchy?
    # TODO implement.
    allowrecursive = False

    allowOneToOne = False
    
    maybederived = True

    @property
    def name(self):
        # ColumnDeriver.Base => Base
        return self.__module__.split('.')[1]
        
    def __init__(self):
        # print('i am initted!', self.name)
        pass
        
    def IsApplicable(self, col):
        return True
        
    def __str__(self):
        n =             'Name : ' + self.name + '\n'        
        subclasses =    'Subclasses : ' + ','.join([cls.__name__ for cls in ColumnDeriverBase.__subclasses__()]) + '\n'
        return n + subclasses
        
    # def GetDerivers(self):
        # return self.__subclasses__()
        
           