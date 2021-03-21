import pandas as pd
import scipy as scipy
import numpy as np
import logging
from pprint import pprint


from ColumnDeriver.Base import ColumnDeriverBase
from ColumnDeriver.Len import ColumnDeriverLen
from ColumnDeriver.Abs import ColumnDeriverAbs
from ColumnDeriver.Upper import ColumnDeriverUpper

logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')

from Column import Column



class AddDerivedColumns:

    delimiter='.'
    
    def __init__(self):
        print('init start')
        self.basederiver = ColumnDeriverBase()
        # Get the list of derivers and initialise all of them.       
        self.allderivers =  [d() for d in self.basederiver.GetDerivers()]
        print('init end')
        
    def Process(self, column):
        print('process start')

        assert(isinstance(column, Column))

        derivedcolumns = []
        
        for deriver in self.allderivers:
        
            if deriver.IsApplicable(column) and column.depth <= deriver.maxdepth:   

                print('applying ' + deriver.name + ' to ' + column.name)
                # Apply the function.  
                # TODO handle where the deriver returns > 1 column.
                newcol = deriver.Apply(column)
                
                if newcol is not None:
                    newcol.name = column.name + self.delimiter + deriver.name
                    column.MakeChild(newcol)                    
                    derivedcolumns.append(newcol)                    
                    # Recursively apply further derivations to this derived column.
                    #self.Process(newcol)
        print('process end')
           
        return derivedcolumns    















    def addAbsolute(self):
        if self.column.dtype == 'int64':            
            newname = self.column.name + '.absolute'    
    
            # Make sure we don't add the derived column > 1 times.
            if not newname in self.column.ChildNames():
                lengths = self.column.series.map(abs)                     
                lengths.name = newname                              
                self.column.MakeChild(Column(lengths))
                    
    
