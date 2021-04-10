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
#  numpy_array = mnc.Process(column, 'X' or 'Y')
#    -or-
#  numpy_array = mnc.ProcessColumnSet(columnset)

class MakeNumericColumns:

    defaultNumericers = ['OneHotEncoded', 'LabelEncoded']
    

    def __init__(self):
        self.basenumericer = ColumnNumericerBase()
        self.allnumericers = []   
         
        # Record which numericer we used to transform which column and target, so we can invert them later.
        self.numericerUsed = dict()
        
    def Register(self, nameString):
        #print('AddDerivedColumns Register started')

        m = importlib.import_module('ColumnNumericer.' + nameString)
        cls = getattr(m, 'ColumnNumericer' + nameString)
        # Initialise the class to start it running ready for action.        
        self.allnumericers.append(cls())

    def RegisterDefaultNumericers(self):
        for numericer in self.defaultNumericers:
            self.Register(numericer)
            
    def ProcessColumn(self, column, target='X'):
        
        boolConverted = False
        for numericer in self.allnumericers: 
                
            if not boolConverted and numericer.IsApplicable(column, target):
                numericColumn = numericer.Apply(column)
                assert(isinstance(numericColumn, np.ndarray))
                boolConverted = True
                # Multi-level dict.
                self.numericerUsed[column.name] = { target : numericer }

        # If we didn't convert, unpack to a numpy array and return.
        if not boolConverted:
            numericColumn = column.series.to_numpy()
            
        return numericColumn
        
    def ProcessColumnSet(self, columnset, target='X'):
    
        numpy_arrays = []
        
        for column in columnset.GetAllColumns():
            numpy_arrays.append(self.ProcessColumn(column))
            
        numpy_array_single = np.column_stack(numpy_arrays)

        print('numpy_array_single:')
        print(numpy_array_single)
        return numpy_array_single

    # To invert a converted, we need to know which numericer we used in the first place.
    # if we didn't use one, we must have passed the column back unconverted.
    def Inverse(self, numpy_array, column, target='X'):
    
        if column.name in self.numericerUsed and target in self.numericerUsed[column.name]:
            return self.numericerUsed[column.name][target].Inverse(numpy_array)
        else:
            return numpy_array
        