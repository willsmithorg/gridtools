import pandas as pd
import scipy as scipy
import numpy as np
import logging
from pprint import pprint


from ColumnDeriver.Base import ColumnDeriverBase
from ColumnDeriver.Len import ColumnDeriverLen
from ColumnDeriver.Abs import ColumnDeriverAbs
from ColumnDeriver.Upper import ColumnDeriverUpper
from ColumnDeriver.MinMaxScaler import ColumnDeriverMinMaxScaler
from ColumnDeriver.RobustScaler import ColumnDeriverRobustScaler
from ColumnDeriver.Date import ColumnDeriverDate

logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')

from Column import Column



class AddDerivedColumns:

    delimiter='.'
    
    def __init__(self):
        print('AddDerivedColumns init start')
        self.basederiver = ColumnDeriverBase()
        # Get the list of derivers and initialise all of them.       
        self.allderivers =  {d() for d in self.basederiver.GetDerivers()}
        print('AddDerivedColumns init end: {num} derivers'.format(num=len(self.allderivers)))
        
    def Process(self, column):

        assert(isinstance(column, Column))

        derivedcolumns = []
        
        # print('\t' * column.depth, 'Processing: {num} derivers'.format(num=len(self.allderivers)))
        for deriver in self.allderivers:
            
            # print('\t' * column.depth, 'considering applying ' + deriver.name + ' to ' + column.name)
       
            if deriver.IsApplicable(column) and column.depth <= deriver.maxdepth:   

                # If we don't allow recursive, make sure we're not using a deriver that was already applied
                # on this column somewhere in one of its ancestors.
                if deriver.allowrecursive or deriver not in self.GetParentColDerivers(column): 
                
                    print('\t' * column.depth, 'applying ' + deriver.name + ' to ' + column.name)
                    # Apply the deriver.  It will return a hash of new columns (keyed by name), if it thinks any are needed.  
                    newcols = deriver.Apply(column)                    
                    for name,newcol in newcols.items():
                        # Handle the deriver forgetting to wrap a series up as a column.
                        if isinstance(newcol, pd.Series):
                            newcol=Column(newcol)
                            
                        newcol.name = column.name + self.delimiter + name
                        # Save how we created it.
                        newcol.deriver = deriver
                        column.MakeChild(newcol)                    
                        derivedcolumns.append(newcol)                    
                    
                else:
                    # print('\t' * column.depth, 'applicable but blocked by recursion')
                    pass
            else:
                # print('\t' * column.depth, 'not applicable')
                pass
                
        # Breadth first.  We added what we can.  Only now try to recurse and add further columns. 
        recursivecolumns = []
        for newcolumn in derivedcolumns:              
            #print('\t' * column.depth, 'recursing into', newcolumn.name)
            #print('type:', type(newcolumn))
            recursivecolumns.append(self.Process(newcolumn))
            #print('\t' * column.depth, '...recursed.')                        
        derivedcolumns.append(recursivecolumns)
        
        # Return all added columns from this node downwards, up to the parent.
        return derivedcolumns    

    # Get the derivers of all columns including the parent.  This helps to prevent
    # us adding recursive derivers that are the same as an ancestor (e.g. with UPPER this 
    # is unnecessary).
    
    def GetParentColDerivers(self, column):
        
        #print(column)
        derivers = []
        c = column
        while c is not None and c.depth > 0:
            derivers.append(c.deriver)
            c = c.parent
            
        return derivers
            