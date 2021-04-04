import pandas as pd
import scipy as scipy
import numpy as np
from AddDerivedColumns import AddDerivedColumns
from Column import Column

from cached_property import cached_property
import logging
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')


class ColumnSet:

    def __init__(self, df):
    
        if not isinstance(df, pd.DataFrame):
            raise(TypeError,'df must be a DataFrame not a ' + str(type(df)))
            
        self.inputcolumns = []
        self.derivedcolumns = []            
            
        for col in df:
            #print('column:',col)
            series = df[col]
            #print(type(series))
            self.inputcolumns.append(Column(series))
        #print('...done : ', len(self.inputcolumns))    
        
        #print(self.inputcolumns)
    
    # Explicit copy, so we copy the elements across, and we can Remove() from a ColumnSet without affecting
    # another ColumnSet it's copied from.
    def __copy__(self):
        obj = ColumnSet.__new__(self.__class__)
        obj.inputcolumns =   [ c for c in self.inputcolumns ]
        obj.derivedcolumns = [ c for c in self.derivedcolumns ]
        return obj
        
            
    # Add the derived columns.  
    # Pass in a AddDerivedColumns() class that you've registered
    # some derivers into.
    def AddDerived(self, adc):               
        for col in self.inputcolumns: 
            #print(col.name)
            newcols = adc.Process(col)
            self.derivedcolumns.extend(newcols)
        

    def GetAllColumns(self):
        return self.inputcolumns + self.derivedcolumns
    
    def GetInputColumnNames(self):
        return [ c.name for c in self.inputcolumns ]
 
    # Return the single column that matches this name.  Or return None if there are none that match.
    def GetInputColumn(self, colname):
        matching = [ c for c in self.inputcolumns if c.name == colname ]
        return matching[0] if len(matching) else None
            
        
    def Remove(self, colname):    
        # Remove this column from the input columns.  
        # For the [:] syntax reason https://stackoverflow.com/questions/1207406/how-to-remove-items-from-a-list-while-iterating
        self.inputcolumns[:] = [ c for c in self.inputcolumns if c.name != colname ]
        
        # And any column derived from this, too.
        self.derivedcolumns[:] = [ c for c in self.derivedcolumns if c.ancestor.name != colname ]
               
                
    def __str__(self):
    

        inputcolsdescrip='\tInputColumns: ' + ', '.join(c.name + '(' + (c.ancestor.name if c.ancestor is not None else '-') + ')' for c in self.inputcolumns)
        derivedcolsdescrip='\tDerivedColumns: ' + ', '.join(c.name + '(' + (c.ancestor.name if c.ancestor is not None else '-') + ')' for c in self.derivedcolumns)

        return 'ColumnSet:\n' + inputcolsdescrip + '\n' + derivedcolsdescrip + '\n'
        