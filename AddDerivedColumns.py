import pandas as pd
import scipy as scipy
import numpy as np
import logging
from pprint import pprint
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')

from Column import Column



class AddDerivedColumns:

    def __init__(self, c):
        assert(isinstance(c, Column))
        self.column = c    
        
    def Add(self):
        # Todo create a transformer class, and we register all the transformers then run them in sequence.
        # Maybe we do it recursively.
        
        self.addLength()
        self.addUpper()
        self.addAbsolute()
        
    def addLength(self):
    
        if self.column.dtype == 'object':            
            newname = self.column.name + '.length'
            
            # Make sure we don't add the derived column > 1 times.
            if not newname in self.column.ChildNames():
                lengths = self.column.series.map(len)                        
                lengths.name = newname                              
                self.column.MakeChild(Column(lengths))
            
    
    def addUpper(self):
    
        if self.column.dtype == 'object':            
            newname = self.column.name + '.upper'
            
            # Make sure we don't add the derived column > 1 times.
            if not newname in self.column.ChildNames():
                lengths = self.column.series.str.upper()                     
                lengths.name = newname                              
                self.column.MakeChild(Column(lengths))
                    
    
    def addAbsolute(self):
        if self.column.dtype == 'int64':            
            newname = self.column.name + '.absolute'    
    
            # Make sure we don't add the derived column > 1 times.
            if not newname in self.column.ChildNames():
                lengths = self.column.series.map(abs)                     
                lengths.name = newname                              
                self.column.MakeChild(Column(lengths))
                    
    
