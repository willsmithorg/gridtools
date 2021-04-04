import pandas as pd
import scipy as scipy
import numpy as np
import logging
from pprint import pprint

import importlib
from Column import Column
from ColumnNumericer.Base import ColumnNumericerBase



logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')


# Convert a column or columnset from a mix of numerical and categorical, to only categorical, ready for a ML model to read.
#

# Usage:
#  mnc = MakeNumericColumns()
#
#  mnc.Register('LabelEncoded')
#    -or-
#  mnc.RegisterDefaultNumericers()
#
#  numpy_array = mnc.Process(column)
#    -or-
#  numpy_array = mnc.ProcessColumnSet(columnset)

class MakeNumericColumns:

    defaultNumericers = ['LabelEncoded']
    

    def __init__(self):
        self.basenumericer = ColumnNumericerBase()
        self.allnumericers = []   
         
        
    def Register(self, nameString):
        #print('AddDerivedColumns Register started')

        m = importlib.import_module('ColumnNumericer.' + nameString)
        cls = getattr(m, 'ColumnNumericer' + nameString)
        # Initialise the class to start it running ready for action.        
        self.allnumericers.append(cls())

    def RegisterDefaultNumericers(self):
        for numericer in self.defaultNumericers:
            self.Register(numericer)
            
    def ProcessColumn(self, column):
        
        boolConverted = False
        for numericer in self.allnumericers: 
                
            if not boolConverted and numericer.IsApplicable(column):
                numericColumn = numericer.Apply(column)
                assert(isinstance(numericColumn, np.ndarray))
                boolConverted = True

        # If we didn't convert, unpack to a numpy array and return.
        if not boolConverted:
            numericColumn = column.series.to_numpy()
            
        return numericColumn
        
    def ProcessColumnSet(self, columnset):
    
        numpy_arrays = []
        
        for column in columnset.GetAllColumns():
            numpy_arrays.append(self.ProcessColumn(column))
            
        numpy_array_single = np.stack(numpy_arrays)
        return np.transpose(numpy_array_single)
