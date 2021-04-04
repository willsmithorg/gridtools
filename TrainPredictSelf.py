import pandas as pd
import numpy as np
import scipy
import copy
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from ColumnSet import ColumnSet
from AddDerivedColumns import AddDerivedColumns
from MakeNumericColumns import MakeNumericColumns

class TrainPredictSelf:

    #xgboost_subsample = 0.8
    xgboost_tree_method='auto' # gpu_hist = use gpu.   auto = default.
    numthreads_xgboost = 8
        
    def __init__(self):
        self.adc = AddDerivedColumns()
        self.adc.RegisterDefaultDerivers()        
        self.mnc = MakeNumericColumns()
        self.mnc.RegisterDefaultNumericers()
        
            
    def Train(self, inputdf):
    
        if( not isinstance(inputdf, pd.DataFrame)):
            raise(TypeError,'df must be a DataFrame not a ', type(inputdf)) 
        
        # Convert dataframe to a columnset so we can make all the derived columns.
        # We only want to do this once.
        
        columnset = ColumnSet(inputdf)
        columnset.AddDerived(self.adc)
        
        
        # Loop through each column, removing it then predicting it.
        for colname in columnset.GetInputColumnNames():
            print(colname)
            # Save a copy of the columnset.  We will be deleting bits of it and we we don't want to affect the full one.
            columnset_X = copy.copy(columnset)            
            # Remove this one column.
            columnset_X.Remove(colname)
            # And convert to numpy for learning.
            numpy_X = self.mnc.ProcessColumnSet(columnset_X)
        
            # Get the Y (predicted) column.
            column_Y = columnset.GetInputColumn(colname)
            numpy_Y = self.mnc.ProcessColumn(column_Y)
        
            # print('colname:\n',colname)
            # print('numpy_array_X:\n',numpy_X)
            # print('numpy_array_Y:\n',numpy_Y)
            
            if column_Y.IsCategorical():            
                model = XGBClassifier(tree_method=self.xgboost_tree_method, verbosity=0, nthread=self.numthreads_xgboost, objective='reg:logistic')
            else:
                model = XGBRegressor (tree_method=self.xgboost_tree_method, verbosity=0, nthread=self.numthreads_xgboost, objective='reg:squarederror') 
            
            print(np.shape(numpy_X))
            print(np.shape(numpy_Y))
            
            crossvalidate = RepeatedKFold(n_repeats=4, random_state=1)
            scores = cross_val_score(model, numpy_X, numpy_Y,cv=crossvalidate,n_jobs=-1)
            print(scores)
            print(np.mean(scores))
