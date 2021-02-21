import pandas as pd
import scipy as scipy
import random
import numpy as np
from TrainPredict import TrainPredict

import logging
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')


class SpotErrors:

    def __init__(self):

        # Constants
        self.std_for_single_prediction_labelencoded = 0.3
        self.zscore_for_error = 5
        
        self.predictedmeans = None
        self.predictedstds = None
        self.boolerrors = None
               

    def Spot(self, trainPredict, predictedmeans, predictedstds, colsname, singlerowid = None):
    
        self.tp = trainPredict
        self.predictedmeans = predictedmeans
        self.predictedstds  = predictedstds
        
        # We either predicted the whole of the converted dataframe, or just one row.
        self.singlerowid = singlerowid
        
        coldnames = self.tp.colmaps2d[colsname]

        
        # Predict, if we haven't already.
        if self.predictedmeans is None or self.predictedstds is None:
            raise(RuntimeError, "Must pass in predicted means and stds to SpotErrors::Spot")

        # Initially, we found no errors.
        if self.singlerowid is None:
            self.boolerrors = pd.DataFrame(False, index=np.arange(self.tp.numrow_predict), columns=coldnames)
        else:
            self.boolerrors = pd.DataFrame(False, index=[0], columns=coldnames)

        for coldname in coldnames:
            for row in range(self.tp.numrow_predict):

                # Are we looking for errors in a array of rows the same size as the training data, or just one row?
                if self.singlerowid is None:
                    cellmean = self.predictedmeans[coldname][row]
                    cellstd  = self.predictedstds[coldname][row]                    
                    actualcellvalue = self.tp.converteddf[coldname][row]
                else:  
                    cellmean = self.predictedmeans[coldname][0]
                    cellstd  = self.predictedstds[coldname][0]
                    actualcellvalue = self.tp.converteddf[coldname][self.singlerowid]
                    
   
                if self.tp.coltyped[coldname] == 'labelencoded':
                    # If prediction <> actual and we are confident about the prediction
                    if cellmean != actualcellvalue and cellstd <= self.std_for_single_prediction_labelencoded:
                        self.__set_boolerror_true(coldname, row)
                elif self.tp.coltyped[coldname] == 'onehot':
                    if round(cellmean) != actualcellvalue:
                        self.__set_boolerror_true(coldname, row)
                else:
                    # 100% confident prediction?
                    if cellstd == 0.0:
                        if cellmean != actualcellvalue:
                            # How different are they?  If substantially different, error.
                            divisor = cellmean if cellmean != 0.0 else actualcellvalue                            
                            difference_as_proportion = np.abs(cellmean - actualcellvalue) / np.abs(divisor)                            
                            # More than 5% different : different.
                            logging.debug('row ' + str(row) + ' mean pred ' + str(cellmean) + ' std ' + str(cellstd) + ' actual ' + str(actualcellvalue) + ' ratio ' + str(difference_as_proportion))

                            if difference_as_proportion > 0.05:
                                self.__set_boolerror_true(coldname, row)
                    else:
                        # Not 100% confident prediction, use the zscore to decide if it's an error.
                        zscore = np.abs(cellmean - actualcellvalue) / cellstd  
                        logging.debug('row ' + str(row) + ' mean pred ' + str(cellmean) + ' std ' + str(cellstd) + ' actual ' + str(actualcellvalue) + ' zscore ' + str(zscore))
                        # If bad, flag it as bad
                        if zscore >= self.zscore_for_error:
                            self.__set_boolerror_true(coldname, row)
                    
        return self.boolerrors
    
    # The shape of the boolean array depends on whether we are predicting the whole table or just 1 row of it.
    def __set_boolerror_true(self, coldname, row):
        if self.singlerowid is None:
            self.boolerrors[coldname][row] = True
        else:
            self.boolerrors[coldname][0] = True
            
            
        
    def PrintErrors(self, sourcedf):
        # Spot errors, if we haven't already.
        if self.boolerrors is None:
            self.SpotErrors(sourcedf)
        
        for colsname in self.tp.sourcedf.columns:
       
            for row in range(self.tp.numrow_predict):
                predicted = None
                stdev = None
 
                # Are we looking for errors in a array of rows the same size as the training data, or just one row?
                if self.singlerowid is None: 
                    actualcellvalue = self.tp.sourcedf[colsname][row]
                else:
                    actualcellvalue = self.tp.sourcedf[colsname][self.singlerowid]
                
                # If it's a literal error, print it.
                if self.coltypes[colsname] == 'raw':
                    coldname = colsname
                    if self.boolerrors[coldname][row]: 
                        predicted = str(self.predictedmeans[coldname][row])
                        stdev     = str(self.predictedstds[coldname][row])
                        
                elif self.tp.coltypes[colsname] == 'labelencoded':
                    coldname = colsname                
                    if self.boolerrors[coldname][row]:                    
                    
                        # What is the predicted value?  If std < 0.3 we are close to a single prediction
                        if self.predictedstds[coldname][row] < self.std_for_single_prediction_labelencoded:
                            predicted = str(self.tp.featuremapd[coldname][round(self.predictedmeans[coldname][row])])
                        else:
                            predicted = '(various)'
                        
                elif self.tp.coltypes[colsname] == 'onehot':
                    # We have to look through all the destination columns, find what this mapped to (could be multiple), and then see
                    # whether we think that's an error.
                   
                    # Which cell did we originally think it is?  Unless we think that's an error, no need to go further.
                    coldname = self.tp.colmaps2d[colsname][self.tp.featuremaps[colsname][actualcellvalue]]
                    
                    if self.boolerrors[coldname][row]:                    
                        predicted=[]
                        for coldname in self.tp.colmaps2d[colsname]:
                        
                            cellmean = self.predictedmeans[coldname][row]
                            cellstd  = self.predictedstds[coldname][row]
                            if round(cellmean) == 1.0:
                                predicted.append(str(self.tp.featuremapd[coldname]))
                        
                        if len(predicted) > 0:
                            if len(predicted) > 1:
                                predicted = '(any one of ' + ','.join(map(str,predicted)) + ')'
                            else:
                                predicted = str(predicted[0])
                        else:
                            predicted = '(no clear prediction)'

                if predicted is not None:
                    logging.debug('row ' + str(row) + ' column ' + colsname  + 
                    ': actual=' + str(actualcellvalue) +
                    ' predicted=' + predicted + 
                    (' stdev='+ stdev if stdev is not None else ''))              


                    
                                 
        
