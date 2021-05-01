import pandas as pd
import numpy as np
import scipy
import copy
import time
import logging
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedKFold
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
        self.adc.RegisterDefaultDerivers()        
        self.mnc = MakeNumericColumns()
        self.mnc.RegisterDefaultNumericers()
        
        self.inputdf = None
        
            
    # Trains each column of a dataframe in turn.  We predict the column using the other columns.
    # Returns a list of the possible labels for the column, and the probability associated with each label, for every cell in
    #   the column.
    def Train(self, inputdf):
    
        if( not isinstance(inputdf, pd.DataFrame)):
            raise(TypeError,'df must be a DataFrame not a ', type(inputdf)) 
        
        self.inputdf = inputdf
        # Convert dataframe to a columnset so we can make all the derived columns.
        # We only want to do this once.        
        columnset = ColumnSet(self.inputdf)
        columnset.AddDerived(self.adc)
        
        results_labels = dict()    
        results_proba = dict()
        
        totaltime = 0
        # Loop through each column, removing it then predicting it.
        for colname in columnset.GetInputColumnNames():
            print('***',colname,'***')
            # Save a copy of the columnset.  We will be deleting bits of it and we we don't want to affect the full one.
            columnset_X = copy.copy(columnset)            
            # Remove this one column (and it's derived columns).
            columnset_X.Remove(colname)
            # And convert to numpy for learning.
            numpy_X = self.mnc.ProcessColumnSet(columnset_X, 'X')
        
            # Get the Y (predicted) column.
            column_Y = columnset.GetInputColumn(colname)
            numpy_Y = self.mnc.ProcessColumn(column_Y, 'Y')

            #print(column_Y.series)
            
            crossvalidate = KFold(n_splits=self.GetOptimalSplits(column_Y), shuffle=True) 

            if column_Y.IsCategorical():   
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
                    
                    print(numpy_X)
                    print(numpy_Y)
                    prediction_proba = cross_val_predict(model, numpy_X, numpy_Y,cv=crossvalidate,n_jobs=self.cross_validation_numthreads, method='predict_proba')
                        
                    # We return the labels associated with 0...n possibles, so that the probabilities can be later joined with the labels
                    # to understand what we actually predicted.
                    labels = np.arange(prediction_proba.shape[1])
                    labels = self.mnc.Inverse(labels, column_Y, 'Y')
                    
                    results_labels[colname] = labels
                    results_proba[colname] = prediction_proba
                    
                             
            else:
                error('This path should not run - everything is a classifier due to KBinsDiscretization')                            

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

    # Get the prediction with the highest probability for each cell.
    # Return it back as 2 DataTables : the first is the prediction, the 2nd is the probability of the prediction.
    def SinglePredictionPerCell(self, results_labels, results_proba):
    
        dflabels = pd.DataFrame()
        dfprobas = pd.DataFrame()
        # For each column, get the index of the highest probability.
        for colname, proba in results_proba.items():            
            max_indices = np.argmax(proba, axis=1)            
            prediction_labels = results_labels[colname][max_indices]
            prediction_proba  = np.max(results_proba[colname], axis=1)
            
            dflabels[colname] = prediction_labels
            dfprobas[colname] = prediction_proba
            
        return dflabels, dfprobas
        
    # Display the probability percentage difference between the top 1 and the 2nd top percentages.
    def Confidence(self, results_proba):
        dfconfidence = pd.DataFrame()
        
        for colname, proba in results_proba.items():
            #print(colname)
            #print(proba)
            proba_sorted = np.sort(proba, axis=1)
            # We define confidence as the difference between the 1st and the 2nd highest probabilities.
            confidence = proba_sorted[:,-1]-proba_sorted[:,-2]
            dfconfidence[colname] = confidence
            
        return dfconfidence


    # These are the differences when we are pretty sure that the prediction is accurate, and it's different
    # from observed.
    def BoolDifferencesConfidentPredictionCorrect(self, results_labels, dflabels, results_proba, dfconfidence):
    
        boolDiff = pd.DataFrame()
        for colname in dflabels:
        
            # Keep tightening the confidence we expect in the top value 
            # until we are getting <= 5% predicted-wrong errors.  Otherwise we're just 
            # spamming the column with errors.
            proportion = 1
            threshold = 0.7
            while proportion > 0.05 and threshold < 0.95:
                boolDiff[colname] = dflabels[colname].ne(self.inputdf[colname]) & dfconfidence[colname].gt(threshold) 
                proportion = np.mean(boolDiff[colname].astype(int))
                threshold += 0.05
                
            
        return boolDiff

    # These are the differences when we are sure the observed is wrong, but are not sure what the correct value is.    
    def BoolDifferencesConfidentObservedWrong(self, results_labels, dflabels, results_proba, dfconfidence):
    
        boolDiff = pd.DataFrame()

        # We have to again map the input data to a numeric array, so we can index the correct column
        # of the results_proba, which contains the probability of each possible class of each row of the particular column.
        for colname in dflabels:
            print(colname)
            column = Column(self.inputdf[colname])
            numericCol = self.mnc.ProcessColumn(column, 'Y').astype(int)
            #print(numericCol)
            probs_for_observed = results_proba[colname][np.arange(results_proba[colname].shape[0]), numericCol]
            print(probs_for_observed)
            # We don't want too many cells to fail on a given column.
            # So take the 2nd percentile on a given column or 20% probability of accuracy, whichever is lower.
            secondpercentile = np.percentile(probs_for_observed, 2)
            #print(firstpercentile)
            threshold = np.min([secondpercentile, 0.2]) 
            
            boolDiff[colname] = probs_for_observed < threshold
            


        return boolDiff
        
    def boolDifferencesCombined(self, results_labels, dflabels, results_proba, dfconfidence):
    
        a = self.BoolDifferencesConfidentObservedWrong    (results_labels, dflabels, results_proba, dfconfidence)
        b = self.BoolDifferencesConfidentPredictionCorrect(results_labels, dflabels, results_proba, dfconfidence)
        
        return a | b
        