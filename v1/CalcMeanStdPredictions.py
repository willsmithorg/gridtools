import unittest
import pandas as pd
import numpy as np
import scipy
from TrainPredict import TrainPredict

import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')



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
            logging.debug(ytest.shape)
            logging.debug(ytest[cold,:,:].shape)
            self.predictedmeans[:,cold], self.predictedstds[:,cold] = self._CalcMeanAndDeviation(ytest[cold,:,:], self.tp.coltyped[coldname])


        # Convert to dataframes with headers, easier for subsequent processing
        self.predictedmeans = pd.DataFrame(data=self.predictedmeans, columns=coldnames)
        self.predictedstds  = pd.DataFrame(data=self.predictedstds,  columns=coldnames)
           
        return (self.predictedmeans, self.predictedstds)           

    def _CalcMeanAndDeviation(self, ypredictions_single_col, coltype):

        logging.debug('calculating mean and stdev for ' + str(ypredictions_single_col) + ' ' + str(coltype))
        
        # If the column is a labelencoded, just calculate the standard deviation of boolean difference from the most common value
        if coltype == 'labelencoded':
            # Calculate the most common value
            if isinstance(ypredictions_single_col, list) or len(ypredictions_single_col.shape) == 1:
                logging.debug('coltype = ' + coltype)
                mn = scipy.stats.mode(ypredictions_single_col, axis=0).mode
                logging.debug('mean shape : ')
                logging.debug(str(mn.shape))
                # And calculate the variation as mean deviation away from that, in boolean terms (different = 1, same = 0)
                std = np.mean(ypredictions_single_col != mn)
            else:
                assert(False, "weird")
                mn = scipy.stats.mode(ypredictions_single_col, axis=0).mode
                std = np.mean(ypredictions_single_col != mn, axis=0)
                mn = mn.reshape(-1)
        else:
            logging.debug('coltype = ' + coltype)

            # Otherwise, conventional mean and standard deviation.   This also works for onehot because we only
            # do 1 column at a time.
            mn = np.mean(ypredictions_single_col, axis=0, keepdims=True) 
            # mn = scipy.stats.mode(ypredictions_single_col).mode

            # Correct because the error cell            
            logging.debug('y predictions:')
            logging.debug(ypredictions_single_col)
            std  = np.std(ypredictions_single_col, axis=0, keepdims=True)
            
        logging.debug('mean is shape ' + str(mn.shape))
        logging.debug('std is shape ' + str(std.shape))
        return (mn, std)
 
