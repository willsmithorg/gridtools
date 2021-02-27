import pandas as pd
import scipy as scipy
import numpy as np

from TrainPredict import TrainPredict
from CalcMeanStdPredictions import CalcMeanStdPredictions
from SpotErrors import SpotErrors

import logging
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')





class Explain:


    def __init__(self):
    
        self.confidence_to_keep_column = 0.15





    def Explain(self, sourcedf, colsname):
    
        # First pass through the data - look for errors in every row.
        tp = TrainPredict()
        ytest = tp.Predict(sourcedf, colsname)
               
        cms = CalcMeanStdPredictions()
        means,stds = cms.Calc(tp, ytest, colsname)               
    
        se = SpotErrors()
        boolerrord = se.Spot(tp, means, stds, colsname)   
        boolerrors, predictions = se.GetErrorsAndPredictions(colsname)
        
        print(boolerrors)
        print('First pass : found ' , str(boolerrors.values.sum()),  ' errors in column ' , colsname)
        
        # Iterate through the errors and relearn a model predicting just this row.
        for row in np.arange(len(boolerrors)):
            
            if boolerrors[colsname][row]:
                print('2nd pass : row=' + str(row))               
                tp = TrainPredict()
                ytest = tp.Predict(sourcedf, colsname, singlerowid = row)
                cms = CalcMeanStdPredictions()
                means,stds = cms.Calc(tp, ytest, colsname) 
                se = SpotErrors()
                boolerrord = se.Spot(tp, means, stds, colsname, singlerowid = row)   
                boolerrors, predictions = se.GetErrorsAndPredictions(colsname, singlerowid = row)               
                
                print(boolerrors)
                print(predictions)
                
                # TODO we shouldn't do this, learned_cols is per coldname not colsname.
                print(tp.learned_cols[colsname])
                # TODO we shouldn't do this, parameter is coldname not colsname.

                print(self.GetBestColumnsToPredict(tp, colsname))
                

    # Todo we shuold accumulate this per colsname.  If 4 coldnames were slightly useful that makes the colsname very useful.
    def GetBestColumnsToPredict(self, tp, coldname):

        coldnames_learnt = tp.learned_cols[coldname]
        
        # Take the average feature importance across all models for each column.
        # If it's high, report this as a useful column.
        bestcoldnames = []
        for coldid in range(len(coldnames_learnt)):
            total_confidence = 0.0 
            for m in range(tp.models_for_confidence):
                total_confidence += tp.models[coldname][m].feature_importances_[coldid]            
            total_confidence /= tp.models_for_confidence

            print('coldname', coldnames_learnt[coldid], ' fi: ', total_confidence)
            
            if total_confidence >= self.confidence_to_keep_column:
                bestcoldnames.append(coldnames_learnt[coldid])
        
        if len(bestcoldnames):
            logging.debug('Most useful columns to predict ''' + coldname + ' were ''' + str(bestcoldnames))
        
        return bestcoldnames
  

        
        
