import pandas as pd
import numpy as np
import copy
import logging
import warnings
# from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold
# from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.svm import SVR
from xgboost import XGBClassifier
# from xgboost import XGBRegressor
from Column import Column
from ColumnSet import ColumnSet
from AddDerivedColumns import AddDerivedColumns
from MakeNumericColumns import MakeNumericColumns

logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')


class TrainPredictSelf:
   
    # The below 2 were tuned for the '06_TrainPredict.py' dataset.
    xgboost_row_subsample=1  # If we set this to <1 we get warnings about LabelEncoder.
    xgboost_col_subsample=0.7
    xgboost_tree_method='auto' # gpu_hist = use gpu.   auto = default.
    max_k_splits = 8 # Don't use more than this number of k-fold splits, even for large datasets.
    regression_loops = 10 # Run this many loops to get a decent mean/stdev for regression columns.
    
    xgboost_numthreads = 1
    cross_validation_numthreads = 8
        
    def __init__(self):
        self.adc = AddDerivedColumns()
        #self.adc.RegisterDefaultDerivers()        
        self.mnc = MakeNumericColumns()
        self.mnc.RegisterDefaultNumericers()
        
        self.observeddf = None
        self.columnset = None # We save this for use in TrainPredictSingleCell because it's expensive 
                              # (due to AddDerived).
            
    # Trains each column of a dataframe in turn.  We predict the column using the other columns.
    # Returns a list of the possible labels for the column, and the probability associated with each label, for every cell in
    #   the column.
    def Train(self, observeddf):
    
        if( not isinstance(observeddf, pd.DataFrame)):
            raise(TypeError,'df must be a DataFrame not a ', type(observeddf)) 
        
        self.observeddf = observeddf
        # Convert dataframe to a columnset so we can make all the derived columns.
        # We only want to do this once.        
        self.columnset = ColumnSet(self.observeddf)
        self.columnset.AddDerived(self.adc)
        
        results_labels = dict()    
        results_proba = dict()
        
        # Loop through each column, removing it then predicting it.
        for colname in self.columnset.GetInputColumnNames():
            # print('***',colname,'***')
            # Save a copy of the columnset.  We will be deleting bits of it and we we don't want to affect the full one.
            columnset_X = copy.copy(self.columnset)            
            # Remove this one column (and it's derived columns).
            columnset_X.Remove(colname)
            # And convert to numpy for learning.
            numpy_X = self.mnc.ProcessColumnSet(columnset_X, 'X')
        
            # Get the Y (predicted) column.
            column_Y = self.columnset.GetInputColumn(colname)
            numpy_Y = self.mnc.ProcessColumn(column_Y, 'Y')

            #print(column_Y.series)
            
            crossvalidate = KFold(n_splits=self.GetOptimalSplits(column_Y), shuffle=True) 

            if not column_Y.IsCategorical():               
                error('This path should not run - everything is a classifier due to KBinsDiscretization')             
            # print('mode : classifier')
            objective = 'binary:logistic' if column_Y.IsBinary() else 'reg:logistic'
                        
            with warnings.catch_warnings():
                warnings.filterwarnings(action="ignore")
                model = XGBClassifier(tree_method=self.xgboost_tree_method, 
                                      nthread=self.GetOptimalXGBoostThreads(column_Y),
                                      objective=objective, 
                                      subsample=self.xgboost_row_subsample, 
                                      use_label_encoder=True,
                                      eval_metric='logloss',                                      
                                      colsample_bytree=self.xgboost_col_subsample)
                
                #print(numpy_X)
                #print(numpy_Y)
                prediction_proba = cross_val_predict(model, numpy_X, numpy_Y,cv=crossvalidate,n_jobs=self.cross_validation_numthreads, method='predict_proba')
                    
                # We return the labels associated with 0...n possibles, so that the probabilities can be later joined with the labels
                # to understand what we actually predicted.
                labels = np.arange(prediction_proba.shape[1])
                labels = self.mnc.Inverse(labels, column_Y, 'Y')
                
                results_labels[colname] = labels
                results_proba[colname] = prediction_proba
                    
                             
                           
        # Return the labels, for each column, for all the possible labels in that column.
        #   results_labels = [ columns][labels ]
        # And return, for each cell, the probability of each label.
        #   results_proba  = [ columns][ rows * labels ]
        return results_labels, results_proba

    # How many splits in k-fold training.  
    # Too many : takes ages on a large dataset.
    # Too few  : low quality on a small dataset because we are training on a smaller proportion of the data.
    def GetOptimalSplits(self, column):    
        # uniques = sum(column.series == val for val in list(column.series))
        # print(uniques)
        
        if column.size > self.max_k_splits:
            return self.max_k_splits
        else:
            return column.size

    # Heuristic  from testing.  If it's a small dataset, there's an overhead to doing multithreaded.       
    def GetOptimalXGBoostThreads(self, column):
        if column.size > 300:
            return 4
        else:
            return 1


    def TrainPredictSingleCell(self, colname, rownum):

        if( self.observeddf is None or self.columnset is None):
            raise(RuntimeError,'You should call TrainPredict before calling TrainPredictSingleCell')

        results_labels = dict()    
        results_proba = dict()
        results_feature_importances = dict()

        # Prep X and Y as in TrainPredict
        # Save a copy of the columnset.  We will be deleting bits of it and we we don't want to affect the full one.
        columnset_X = copy.copy(self.columnset)            
        # Remove this one column (and it's derived columns).
        columnset_X.Remove(colname)
        # And convert to numpy for learning.
        numpy_X = self.mnc.ProcessColumnSet(columnset_X, 'X')
        # print(numpy_X)
    
        # Get the Y (predicted) column.
        column_Y = self.columnset.GetInputColumn(colname)
        numpy_Y = self.mnc.ProcessColumn(column_Y, 'Y')
        
        # Now split out the prediction row.
        predict_numpy_X = numpy_X[rownum,:].reshape(1,-1)               
        # print(predict_numpy_X.shape)
        
        # And remove from training.
        train_numpy_X = np.delete(numpy_X, rownum, 0)
        train_numpy_Y = np.delete(numpy_Y, rownum, 0)        
        # print(train_numpy_X.shape)
        # print(train_numpy_Y.shape)
        
        if not column_Y.IsCategorical():               
            error('This path should not run - everything is a classifier due to KBinsDiscretization')             
        # print('mode : classifier')
        objective = 'binary:logistic' if column_Y.IsBinary() else 'reg:logistic'
               
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore")
            model = XGBClassifier(tree_method=self.xgboost_tree_method, 
                                  nthread=self.GetOptimalXGBoostThreads(column_Y),
                                  objective=objective, 
                                  subsample=self.xgboost_row_subsample, 
                                  use_label_encoder=True,
                                  eval_metric='logloss',                                      
                                  colsample_bytree=self.xgboost_col_subsample)
            
            model.fit(train_numpy_X, train_numpy_Y)

            prediction_proba = model.predict_proba(predict_numpy_X)
                
            # We return the labels associated with 0...n possibles, so that the probabilities can be later joined with the labels
            # to understand what we actually predicted.
            labels = np.arange(prediction_proba.shape[1])
            labels = self.mnc.Inverse(labels, column_Y, 'Y')
            
            results_labels[colname] = labels
            results_proba[colname] = prediction_proba     
            # print(columnset_X.GetAllColumnNames())
            # print(model.feature_importances_)
            
            # add a multipler to feature_importances_ so it sums to 1.
            # print(model.feature_importances_)
            
            results_feature_importances[colname] = { k:v for (k,v) in zip(columnset_X.GetAllColumnNames(), model.feature_importances_) if v > 0 }
            
        return results_labels, results_proba, results_feature_importances
