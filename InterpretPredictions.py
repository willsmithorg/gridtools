import pandas as pd
import numpy as np
import logging
import warnings

# from xgboost import XGBRegressor
from Column import Column
from MakeNumericColumns import MakeNumericColumns

logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')


class InterpretPredictions:
   

    def __init__(self):
       
        self.mnc = MakeNumericColumns()
        self.mnc.RegisterDefaultNumericers()
        
 
    # Get the prediction with the highest probability for each cell.
    # Return it back as 2 DataTables : the first is the prediction, the 2nd is the probability of the prediction.
    def SinglePredictionPerCell(self, results_labels, results_proba):
    
        dfpredicted = pd.DataFrame()
        dfprobas = pd.DataFrame()
        # For each column, get the index of the highest probability.
        for colname, proba in results_proba.items():            
            max_indices = np.argmax(proba, axis=1)            
            prediction_labels = results_labels[colname][max_indices]
            prediction_proba  = np.max(results_proba[colname], axis=1)
            
            dfpredicted[colname] = prediction_labels
            dfprobas[colname] = prediction_proba
            
        return dfpredicted, dfprobas
        
    # Display the probability percentage difference between the top 1 and the 2nd top percentages.
    # We consider this to be how confident we are.
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

    # Return a boolean array of the instances where we think the prediction is wrong.
    # In some cases we are confident that the observed is wrong, but don't have a strong prediction.
    # In other cases, we have a strong prediction 
    def boolDifferences(self, observeddf, results_labels, results_proba):
    
        dfconfidence = self.Confidence(results_proba)
        dfpredicted, dfprobas = self.SinglePredictionPerCell(results_labels, results_proba)
        a = self._BoolDifferencesConfidentObservedWrong    (observeddf, results_labels, dfpredicted, results_proba, dfconfidence)
        b = self._BoolDifferencesConfidentPredictionCorrect(observeddf, results_labels, dfpredicted, results_proba, dfconfidence)
        
        return a | b
        

    # These are the differences when we are pretty sure that the prediction is accurate, and it's different
    # from observed.
    def _BoolDifferencesConfidentPredictionCorrect(self, observeddf, results_labels, dfpredicted, results_proba, dfconfidence):
    
        boolDiff = pd.DataFrame()
        for colname in dfpredicted:
        
            # Keep tightening the confidence we expect in the top value 
            # until we are getting <= 5% predicted-wrong errors.  Otherwise we're just 
            # spamming the column with errors.
            proportion = 1
            threshold = 0.7
            while proportion > 0.05 and threshold < 0.95:
                # Predicted <> Observed AND confident in predicted.
                boolDiff[colname] = dfpredicted[colname].ne(observeddf[colname]) & dfconfidence[colname].gt(threshold) 
                proportion = np.mean(boolDiff[colname].astype(int))
                threshold += 0.05
                
            
        return boolDiff

    # These are the differences when we are sure the observed is wrong, but are not sure what the correct value is.    
    def _BoolDifferencesConfidentObservedWrong(self, observeddf, results_labels, dfpredicted, results_proba, dfconfidence):
    
        boolDiff = pd.DataFrame()

        # We have to again map the input data to a numeric array, so we can index the correct column
        # of the results_proba, which contains the probability of each possible class of each row of the particular column.
        for colname in dfpredicted:
            #print(colname)
            column = Column(observeddf[colname])
            numericCol = self.mnc.ProcessColumn(column, 'Y').astype(int)
            #print(numericCol)
            probs_for_observed = results_proba[colname][np.arange(results_proba[colname].shape[0]), numericCol]
            #print(probs_for_observed)
            # We don't want too many cells to fail on a given column.
            # So take the 2nd percentile on a given column or 20% probability of accuracy, whichever is lower.
            secondpercentile = np.percentile(probs_for_observed, 2)
            #print(firstpercentile)
            threshold = np.min([secondpercentile, 0.2]) 
            
            boolDiff[colname] = probs_for_observed < threshold

        return boolDiff
        
