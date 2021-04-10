import pandas as pd
import numpy as np
import scipy
import copy
import time
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBClassifier, XGBRegressor
from ColumnSet import ColumnSet
from AddDerivedColumns import AddDerivedColumns
from MakeNumericColumns import MakeNumericColumns

class TrainPredictSelf:
   
    # The below 2 were tuned for the '06_TrainPredict.py' dataset.
    xgboost_row_subsample=1  # If we set this to <1 we get warnings about LabelEncoder.
    xgboost_col_subsample=0.5
    xgboost_tree_method='auto' # gpu_hist = use gpu.   auto = default.
    max_k_splits = 20 # Don't use more than this number of k-fold splits, even for large datasets.
    regression_loops = 10 # Run this many loops to get a decent mean/stdev for regression columns.
    
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
        
        results = dict()        
        
        totaltime = 0
        # Loop through each column, removing it then predicting it.
        for colname in columnset.GetInputColumnNames():
            # print('***',colname,'***')
            # Save a copy of the columnset.  We will be deleting bits of it and we we don't want to affect the full one.
            columnset_X = copy.copy(columnset)            
            # Remove this one column (and it's derived columns).
            columnset_X.Remove(colname)
            # And convert to numpy for learning.
            numpy_X = self.mnc.ProcessColumnSet(columnset_X, 'X')
        
            # Get the Y (predicted) column.
            column_Y = columnset.GetInputColumn(colname)
            numpy_Y = self.mnc.ProcessColumn(column_Y, 'Y')


            crossvalidate = KFold(n_splits=self.GetOptimalSplits(column_Y)) 

            if column_Y.IsCategorical():   
                # print('mode : classifier')
                objective = 'binary:logistic' if column_Y.IsBinary() else 'reg:logistic'
                
                model = XGBClassifier(tree_method=self.xgboost_tree_method, 
                                      nthread=self.xgboost_numthreads, 
                                      objective=objective, 
                                      subsample=self.xgboost_row_subsample, 
                                      eval_metric='logloss',                                      
                                      colsample_bytree=self.xgboost_col_subsample)
                
                results[colname] = cross_val_predict(model, numpy_X, numpy_Y,cv=crossvalidate,n_jobs=self.cross_validation_numthreads, method='predict_proba')
                             
            else:
                # print('mode : regressor')     
                # For a regressor, we train and run the model multiple times to get
                # an array of predictions that we can later derive mean and std() from.
                predictions = []
                for i in range(self.regression_loops):

                    model = XGBRegressor (tree_method=self.xgboost_tree_method, 
                                          nthread=self.xgboost_numthreads, 
                                          objective='reg:squarederror', 
                                          subsample=self.xgboost_row_subsample, 
                                          colsample_bytree=self.xgboost_col_subsample,
                                          random_state = i)                 # Make sure xgboost produces different predictions every time.

                        
                    p = cross_val_predict(model, numpy_X, numpy_Y,cv=crossvalidate,n_jobs=self.cross_validation_numthreads, method='predict')
                    predictions.append(p)
                results[colname] = predictions                             

        return results

    # How many splits in k-fold training.  
    # Too many : takes ages on a large dataset.
    # Too few  : low quality on a small dataset because we are training on a smaller proportion of the data.
    def GetOptimalSplits(self, columnset):
        if columnset.size > self.max_k_splits:
            return self.max_k_splits
        else:
            return columnset.size
        
        
