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
    
        self.confidence_to_keep_column = 0.25





    def Explain(self, sourcedf, colsname):
    
        # First pass through the data - look for errors in every row.
        tp1 = TrainPredict()
        ytest = tp1.Predict(sourcedf, colsname)
               
        cms = CalcMeanStdPredictions()
        means,stds = cms.Calc(tp1, ytest, colsname)               
    
        se = SpotErrors()
        boolerrord_pass1 = se.Spot(tp1, means, stds, colsname)   
        boolerrors_pass1, predictions_pass1 = se.GetErrorsAndPredictions(colsname)
        
        print(boolerrors_pass1)
        print('First pass : found ' , str(boolerrors_pass1.values.sum()),  ' errors in column ' , colsname)
        
        # Iterate through the errors and relearn a model predicting just for this row.  This gives us predicted columns that 
        # explain this particular cell.
        for row in np.arange(len(boolerrors_pass1)):
            
            if boolerrors_pass1[colsname][row]:
                print('2nd pass : row=' + str(row))               
                tp2 = TrainPredict()
                ytest = tp2.Predict(sourcedf, colsname, singlerowid = row)
                cms = CalcMeanStdPredictions()
                means,stds = cms.Calc(tp2, ytest, colsname) 
                se = SpotErrors()
                boolerrord_pass2 = se.Spot(tp2, means, stds, colsname, singlerowid = row)   
                boolerrors_pass2, predictions_pass2 = se.GetErrorsAndPredictions(colsname, singlerowid = row)               
                
                if boolerrord_pass2.shape != (1,1):
                    raise(RuntimeError, 'boolerrors_pass2 should be only 1 row and 1 col on 2nd pass, not ', str(boolerrord_pass2.shape))
                if predictions_pass2.shape != (1,1):
                    raise(RuntimeError, 'predictions_pass2 should be only 1 row and 1 col on 2nd pass, not ', str(predictions_pass2.shape))                    
                    
                if boolerrors_pass2[colsname][0]:
                    print('Still bad on pass 2')
                else:   
                    print('No longer bad on pass 2')
                    
                logging.debug(boolerrors_pass2)
                logging.debug(predictions_pass2)
                
                # Find out how likely is actual.  
                if tp2.coltypes[colsname] == 'labelencoded' or tp2.coltypes[colsname] == 'actual':
                    bestCols = self.GetBestColumnsToPredict(tp2, colsname)  

                    # Get percentage of rows that had actual and predicted.
                    actual=sourcedf[colsname][row]
                    if len(predictions_pass2[colsname][0]):                    
                        prediction1=predictions_pass2[colsname][0][0]
                    else:
                        prediction1=None
                        
                    print('actual/prediction')
                    print(actual)
                    print(prediction1)
                    
                    total_rows,actual_rows,pred_rows = self._CalcPercentageOfRows(sourcedf, colsname, bestCols, row, actual, prediction1)
                    print('Row', str(row), 'col', str(colsname), 'is', str(actual), '(', str(actual_rows), 'rows seen).  We think it should be', str(prediction1), '(', 
                           str(pred_rows), 'rows seen) of', str(total_rows), 'total rows where')
                    for (c,v) in self._GetPredictionColsVals(sourcedf, bestCols, row):
                        print('column ', str(c), 'is', str(v) )
                           
                else:    
                    raiseError(ValueError, 'onehot not yet handled')
                    


    # Todo we shuold accumulate this per colsname.  If 4 coldnames were slightly useful that makes the colsname very useful.
    def GetBestColumnsToPredict(self, tp, coldname):

        print('GetBestColumnsToPredict for ' + coldname)
        coldnames_learnt = tp.learned_cols[coldname]
        colsnames_learnt = set()
        best_colsnames = []
        
        for name in coldnames_learnt:
            colsnames_learnt.add(tp.colmapd2s[name])
        
        logging.debug('coldnames_learnt ' + str(coldnames_learnt))
        logging.debug('colsnames_learnt ' + str(colsnames_learnt))
        
        # Take the average feature importance across all models for each columnd.
        # Then accumulate them into columns's.
        
        # If it's high, report this as a useful column.
        bestcoldnames = []       
        average_confidence_per_coldname = dict()
        average_confidence_per_colsname = dict()
        
        for coldid in range(len(coldnames_learnt)):
            coldname_learnt = coldnames_learnt[coldid]            
            average_confidence_per_coldname[coldname_learnt] = np.mean([tp.models[coldname][m].feature_importances_[coldid] for m in np.arange(tp.models_for_confidence)])
            logging.debug('coldname', coldnames_learnt[coldid], ' fi: ', average_confidence_per_coldname[coldname_learnt])
        
        # Accumulate the feature importances by colsname.
        for colsname_learnt in colsnames_learnt:
            average_confidence_per_colsname[colsname_learnt] = np.sum([average_confidence_per_coldname[x] for x in tp.colmaps2d[colsname_learnt]])
            print('colsname', colsname_learnt, ' fi: ', average_confidence_per_colsname[colsname_learnt])
            if average_confidence_per_colsname[colsname_learnt] >= self.confidence_to_keep_column:
                best_colsnames.append(colsname_learnt)
                
        return best_colsnames

        
        
        
    def _CalcPercentageOfRows(self, df, colsname, predictioncols, row, eqvalue1, eqvalue2):
    

        assert(isinstance(predictioncols, list))
        assert(isinstance(colsname, str))
        
        # Throw away any cols except those we need, to increase performance of the below.
        df = df[predictioncols + [colsname]]
        logging.debug(df)
        
        # How many match?  First let's chop out the row itself.
        # df_filtered = df.drop(index=row)   # I don't think we need to do this.
        df_filtered = df
        for c,v in self._GetPredictionColsVals(df_filtered, predictioncols, row):
            logging.debug(c,v)
            df_filtered = df_filtered[df_filtered[c]==v]
                
        df_filtered_matching1 = df_filtered[df_filtered[colsname]==eqvalue1]
        df_filtered_matching2 = df_filtered[df_filtered[colsname]==eqvalue2]
        
        return (len(df_filtered), len(df_filtered_matching1), len(df_filtered_matching2))
        
        
    # Returns a zipped list of predicted columns and predicted values for this row, given just the prediction columns and the row number.
    def _GetPredictionColsVals(self, df, predictioncols, row):
        predictionvals = []
        for col in predictioncols:
            # Get the array of actual values at this row.
            predictionvals.append(df[col][row])   
        
        return zip(predictioncols, predictionvals)