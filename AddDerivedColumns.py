import pandas as pd
import scipy as scipy
import numpy as np
import logging
from pprint import pprint

import importlib
from Column import Column
from ColumnDeriver.Base import ColumnDeriverBase



logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')




class AddDerivedColumns:

    delimiter='.'
    defaultDerivers = ['Abs', 
                       'Date', 
                       'Len', 
                       'MinMaxScaler', 
                       'RobustScaler',
# too slow!            'SentenceEmbedder', 
                       'TokenizerCharDecimal', 
                       'Upper' ]
                     
    def __init__(self):
        #print('AddDerivedColumns init start')
        self.basederiver = ColumnDeriverBase()
        self.allderivers = []   
        #print('AddDerivedColumns init end: {num} derivers'.format(num=len(self.allderivers)))        

    def Register(self, nameString):
        #print('AddDerivedColumns Register started')

        m = importlib.import_module('ColumnDeriver.' + nameString)
        cls = getattr(m, 'ColumnDeriver' + nameString)
        # Initialise the class to start it running ready for action.        
        self.allderivers.append(cls())


        
        # self.basederiver = ColumnDeriverBase()
        # # Get the list of derivers and initialise all of them.       
        # self.allderivers =  {d() for d in self.basederiver.GetDerivers()}
        #print(self.basederiver)
        #print('AddDerivedColumns Register end: {num} derivers'.format(num=len(self.allderivers)))

    def RegisterDefaultDerivers(self):
        for deriver in self.defaultDerivers:
            self.Register(deriver)
            
    def Process(self, column):
    
        assert(isinstance(column, Column))

        derivedcolumns = []
        
        # print('\t' * column.depth, 'Processing: {num} derivers'.format(num=len(self.allderivers)))
        for deriver in self.allderivers:
            # print('\t' * column.depth, 'considering applying ' + deriver.name + ' to ' + column.name)
       
            if (column.deriver is None or column.deriver.maybederived) and column.depth <= deriver.maxdepth and deriver.IsApplicable(column) :   

                # If we don't allow recursive, make sure we're not using a deriver that was already applied
                # on this column somewhere in one of its ancestors.
                if deriver.allowrecursive or deriver not in self.GetAncestorsColDerivers(column): 
                
                    # print('\t' * column.depth, 'applying ' + deriver.name + ' to ' + column.name)
                    # Apply the deriver.  It will return a hash of new columns (keyed by name), if it thinks any are needed.  
                    newcols = deriver.Apply(column)                    
                    for name,newcol in newcols.items():
                        # print('\t' * column.depth,': got', name)
                        # Generally derivers don't bother to wrap a pd.Series into a Column, so do it for them.
                        if not isinstance(newcol, Column) and isinstance(newcol, pd.Series):
                            newcol=Column(newcol)

                        # No point adding this column if its cardinality is 1.
                        unique_elements = newcol.nunique
                        # print('\t' * column.depth,':', name, 'has', unique_elements, 'unique elements')
                        if unique_elements == 1:
                            # print('no need to add', name, 'based on',column.name, 'it results in a column with cardinality == 1')
                            pass            
                        elif not deriver.allowOneToOne and column.IsOneToOne(newcol):                            
                            # print('no need to add', name, 'based on',column.name, 'because they are 1:1')
                            pass
                        else:
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
        newchildcols = []
        for newcolumn in derivedcolumns:              
            #print('\t' * column.depth, 'recursing into', newcolumn.name)
            #print('type:', type(newcolumn))
            newchildcols.extend(self.Process(newcolumn))
            #print('\t' * column.depth, '...recursed.')                        
        derivedcolumns.extend(newchildcols)
        
        # Return all added columns from this node downwards, up to the parent.
        return derivedcolumns    


    # Get the derivers of all columns including the parent.  This helps to prevent
    # us adding recursive derivers that are the same as an ancestor (e.g. with UPPER this 
    # is unnecessary).
    
    def GetAncestorsColDerivers(self, column):
        
        #print(column)
        derivers = []
        c = column
        while c is not None and c.depth > 0:
            derivers.append(c.deriver)
            c = c.parent
            
        return derivers
      
