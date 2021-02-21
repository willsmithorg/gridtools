import pandas as pd
import scipy as scipy
import random
import numpy as np
from TrainPredict import TrainPredict



class SpotErrors:

    def __init__(self, trainPredict):

        # Constants
        self.std_for_single_prediction_labelencoded = 0.3
        self.zscore_for_error = 10
        
        self.predictedmeans = None
        self.predictedstds = None
        self.boolerrors = None
               
        self.tp = trainPredict
    
    def SpotErrors(self, singlerowid = None):
    
        # We either predict the whole of the converted dataframe, or just one row.
        self.singlerowid = singlerowid
        
        # Predict, if we haven't already.
        if self.predictedmeans is None:
            ytest = self.Predict(sourcedf, singlerowid)
    
        # Initially, we found no errors.
        if self.singlerowid is None:
            self.boolerrors = pd.DataFrame(False, index=np.arange(len(self.tp.converteddf.index)), columns=self.tp.converteddf.columns)
        else:
            self.boolerrors = pd.DataFrame(False, index=[0], columns=self.tp.converteddf.columns)

      
 

        for coldname in self.tp.converteddf.columns:

            for row in range(self.tp.numrow_predict):

                # Are we looking for errors in a array of rows the same size as the training data, or just one row?
                if self.singlerowid is None:
                    cellmean = self.predictedmeans[coldname][row]
                    cellstd  = self.predictedstds[coldname][row]                    
                    actualcellvalue = self.converteddf[coldname][row]
                else:  
                    cellmean = self.predictmeans[coldname][0]
                    cellmean = self.predictedstds[coldname][0]
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
                            self.__set_boolerror_true(coldname, row)
                    else:
                        # Not 100% confident prediction, use the zscore to decide if it's an error.
                        zscore = np.abs(cellmean - actualcellvalue) / cellstd       
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
                    print('row ' + str(row) + ' column ' + colsname  + 
                    ': actual=' + str(actualcellvalue) +
                    ' predicted=' + predicted + 
                    (' stdev='+ stdev if stdev is not None else ''))              


                    
                                 
        
