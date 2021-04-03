import pandas as pd
import scipy as scipy
import numpy as np
from AddDerivedColumns import AddDerivedColumns
from Column import Column

from cached_property import cached_property
import logging
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')


class ColumnSet:

    inputcolumns = []
    derivedcolumns = []
    
    deriversToUse = ['Abs', 
                     'Date', 
                     'Len', 
                     'MinMaxScaler', 
                     'RobustScaler',
                     'SentenceEmbedder', 
                     'TokenizerCharDecimal', 
                     'Upper' ]
    
    
    def __init__(self, df):
    
        if not isinstance(df, pd.DataFrame):
            raise(TypeError,'df must be a DataFrame not a ' + str(type(df)))
            
            
        for col in df:
            series = df[col]
            #print(type(series))
            self.inputcolumns.append(Column(series))
            
        #print(self.inputcolumns)
    
    
    
    def AddDerived(self):
    
        # Create the AddDerivedColumns processor and register all the derivers.
        adc = AddDerivedColumns()        
        for deriver in self.deriversToUse:
            adc.Register(deriver)
                         
        for col in self.inputcolumns: 
            print(col.name)
            newcols = adc.Process(col)
            self.derivedcolumns.extend(newcols)
        
        for col in self.derivedcolumns:
            print('col:', col.name, ' parent:' , col.parent.name if col.parent is not None else '-')
            print(col)


    def GetAllColumns(self):
        return self.inputcolumns + self.derivedcolumns
        