import pandas as pd
import numpy as np
import scipy
import copy
import time
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from ColumnSet import ColumnSet
from AddDerivedColumns import AddDerivedColumns
from MakeNumericColumns import MakeNumericColumns

class TrainPredictSelf:
   
    # The below 2 were tuned for the '06_TrainPredict.py' dataset.
    xgboost_row_subsample=1  # If we set this to <1 we get warnings about LabelEncoder.
    xgboost_col_subsample=0.5
    xgboost_tree_method='hist' # gpu_hist = use gpu.   auto = default.
    
    
    xgboost_numthreads = 1
    cross_validation_numthreads = -1
        
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
        
        
        totaltime = 0
        # Loop through each column, removing it then predicting it.
        for colname in columnset.GetInputColumnNames():
            print(colname)
            # Save a copy of the columnset.  We will be deleting bits of it and we we don't want to affect the full one.
            columnset_X = copy.copy(columnset)            
            # Remove this one column.
            columnset_X.Remove(colname)
            # And convert to numpy for learning.
            numpy_X = self.mnc.ProcessColumnSet(columnset_X, 'X')
        
            # Get the Y (predicted) column.
            column_Y = columnset.GetInputColumn(colname)
            numpy_Y = self.mnc.ProcessColumn(column_Y, 'Y')
        
            # print('colname:\n',colname)
            # print('numpy_array_X:\n',numpy_X)
            # print('numpy_array_Y:\n',numpy_Y)
            

            if column_Y.IsCategorical():            
                model = XGBClassifier(tree_method=self.xgboost_tree_method, 
                                      nthread=self.xgboost_numthreads, 
                                      objective='reg:logistic', 
                                      subsample=self.xgboost_row_subsample, 
                                      colsample_bytree=self.xgboost_col_subsample)
            else:
                model = XGBRegressor (tree_method=self.xgboost_tree_method, 
                                      nthread=self.xgboost_numthreads, 
                                      objective='reg:squarederror', 
                                      subsample=self.xgboost_row_subsample, 
                                      colsample_bytree=self.xgboost_col_subsample) 
            

            
            crossvalidate = RepeatedKFold(n_repeats=3, random_state=3)
            start = time.time()
            scores = cross_val_score(model, numpy_X, numpy_Y,cv=crossvalidate,n_jobs=self.cross_validation_numthreads)
            end = time.time()
            totaltime += end-start
            #print(scores)
            print(np.mean(scores))
        
        print('Total time', totaltime, 'seconds')
