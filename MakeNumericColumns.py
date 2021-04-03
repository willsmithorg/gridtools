import pandas as pd
import scipy as scipy
import numpy as np
import logging
from pprint import pprint

import importlib
from Column import Column
from ColumnNumericer.Base import ColumnNumericerBase



logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')




class MakeNumericColumns:

    defaultNumericers = ['LabelEncoded']
    

    def __init__(self):
        self.basenumericer = ColumnNumericerBase()
        self.allnumericers = []   
        
        # Register the numericers in the order we want them tried.
        self.Register('LabelEncoded')
        
        
    def Register(self, nameString):
        #print('AddDerivedColumns Register started')

        m = importlib.import_module('ColumnNumericer.' + nameString)
        cls = getattr(m, 'ColumnNumericer' + nameString)
        # Initialise the class to start it running ready for action.        
        self.allnumericers.append(cls())

    def RegisterDefaultNumericers(self):
        for numericer in self.defaultNumericers:
            self.Register(numericer)
            
    def Process(self, column):
        
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
            numpy_arrays.append(self.Process(column))
            
        numpy_array_single = np.stack(numpy_arrays)
        return numpy_array_single
