import unittest
import pandas as pd
import numpy as np
import scipy
from TrainPredict import TrainPredict


class CalcMeanStdPredictions:

    def __init__(self):
	
        # The trained models.
        self.tp = None

    def Calc(self, trainPredict, ytest, colsname):
    
        self.tp = trainPredict

        coldnames = self.tp.colmaps2d[colsname]

        # ytest is (numcold, models_to_average, nrows_trained) in size.
        
        numrow_predict = ytest.shape[2]
        numcold = ytest.shape[0]
        
        # We got the predictions in ytest, several per column.  Average them.
        self.predictedmeans = np.zeros((numrow_predict, numcold))
        self.predictedstds  = np.zeros((numrow_predict, numcold))

        for cold in range(len(coldnames)): 
            coldname = self.tp.colmaps2d[colsname][cold]
            # Accumulate means and standard deviations of the predictions per column.
            #print(ytest.shape)
            #print(ytest[cold,:,:].shape)
            self.predictedmeans[:,cold], self.predictedstds[:,cold] = self._CalcMeanAndDeviation(ytest[cold,:,:], self.tp.coltyped[coldname])


        # Convert to dataframes with headers, easier for subsequent processing
        self.predictedmeans = pd.DataFrame(data=self.predictedmeans, columns=coldnames)
        self.predictedstds  = pd.DataFrame(data=self.predictedstds,  columns=coldnames)
           
        return (self.predictedmeans, self.predictedstds)           

    def _CalcMeanAndDeviation(self, ypredictions_single_col, coltype):
                
        # If the column is a labelencoded, just calculate the standard deviation of boolean difference from the most common value
        if coltype == 'labelencoded':
            # Calculate the most common value
            if isinstance(ypredictions_single_col, list) or len(ypredictions_single_col.shape) == 1:
                mean = scipy.stats.mode(ypredictions_single_col).mode
                # And calculate the variation as mean deviation away from that, in boolean terms (different = 1, same = 0)
                std = np.mean(ypredictions_single_col != mean)
            else:
                assert(False, "weird")
                mean = scipy.stats.mode(ypredictions_single_col, axis=1).mode
                std = np.mean(ypredictions_single_col != mean, axis=1)
                mean = mean.reshape(-1)
        else:
            # Otherwise, conventional mean and standard deviation.   This also works for onehot because we only
            # do 1 column at a time.
            mean = np.mean(ypredictions_single_col, axis=0, keepdims=True)           
            std  = np.std(ypredictions_single_col, axis=0, keepdims=True)
            
                
        return (mean, std)
 