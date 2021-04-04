import pandas as pd
import numpy as np
import scipy
from ColumnSet import ColumnSet
from AddDerivedColumns import AddDerivedColumns
from MakeNumericColumns import MakeNumericColumns

class TrainPredictSelf:

    def __init__(self):
        self.adc = AddDerivedColumns()
        self.adc.RegisterDefaultDerivers()        
        self.mcn = MakeNumericColumns()
        self.mcn.RegisterDefaultNumericers()
        
            
    def TrainPredict(self, inputdf):
    
        if( not isinstance(inputdf, pd.DataFrame)):
            raise(TypeError,'df must be a DataFrame not a ', type(inputdf)))    
        
        # Convert dataframe to a columnset so we can make all the derived columns.
        # We only want to do this once.
        
        cs = ColumnSet(data1)
        cs.AddDerived(self.adc)
        
        
        