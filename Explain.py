import pandas as pd
import scipy as scipy
import random
import numpy as np

import logging
logging.basicConfig(level=logging.INFO datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')



class Explain:


    def __init__(self):
    
        self.confidence_to_keep_column = 0.25



    def GetBestColumnsToPredict(self, coldname):
        if self.boolerrors is None:
            raise(RuntimeError, "Should not call GetBestColumns before running Prediction")
        
        coldnames_learnt = self.__remove_predicted_column_names(self.converteddf.columns.values, coldname)
        
        # Take the average feature importance across all models for each column.
        # If it's high, report this as a useful column.
        bestcoldnames = []
        for coldid in range(len(coldnames_learnt)):
            total_confidence = 0.0 
            for m in range(self.models_for_confidence):
                total_confidence += self.models[coldname][m].feature_importances_[coldid]
            
            total_confidence /= self.models_for_confidence
            
            if total_confidence >= self.confidence_to_keep_column:
                bestcoldnames.append(coldnames_learnt[coldid])
        
        if len(bestcoldnames):
            print('Most useful columns to predict ''' + coldname + ' were ''' + str(bestcoldnames))
            
  
    # Return the list of columns we are left with to predict from, if we are predicting column 'coldname'
    def __remove_predicted_column_names(self, all_coldnames, coldname):
        if self.coltyped[coldname] == 'raw' or self.coltyped[coldname] == 'labelencoded':           
            remaining_coldnames = np.setdiff1d(all_coldnames, coldname)
        elif self.coltyped[coldname] == 'onehot':
            colsname = self.colmapd2s[coldname]
            coldnames_to_remove = self.colmaps2d[colsname]
            remaining_coldnames = np.setdiff1d(all_coldnames, coldnames_to_remove)

        else:
            raise(TypeError,'coltyped must be one of (raw, labelencoded, onehot) not ' + self.coltyped)                            
            
        return remaining_coldnames
        
        
