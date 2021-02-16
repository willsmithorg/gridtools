import pandas as pd
import scipy as scipy
import random
import numpy as np
import sklearn.model_selection as model_selection
from xgboost import XGBClassifier, XGBRegressor, DMatrix
from MakeFrameNumeric import MakeFrameNumeric


# Train and predict a grid of tabular data (either a dict or a dataframe)
# 
# Nomenclature:
#     cold = column in the destination dataframe (labelencoded, onehotencoded)
#     cols = column in the source dataframe      (as passed in by the user)

class TrainPredict:

    def __init__(self):
         

        self.sourcedf = None
        self.converteddf = None
        self.numpydf = None
        self.numrow = 0
        self.numcold = 0  # Number of destination i.e. converted columns.
        self.coltyped = None
        
        self.models_for_confidence = 10
        self.zscore_for_error = 10
        


        self.models = None
        self.predictedmeans = None
        self.predictedstds = None
        self.boolerrors = None
        
        # Make the predictions deterministic
        random.seed(42)
        np.random.seed(42)
        
 
    def Train(self, sourcedf):
    
        if (isinstance(sourcedf, dict)):
            sourcedf = pd.DataFrame(sourcedf)
            
        if( not isinstance(sourcedf, pd.DataFrame)):
            raise(TypeError,'df must be a DataFrame not a ' + str(type(sourcedf)))
        
        self.sourcedf    = sourcedf        
        mfn = MakeFrameNumeric()
        self.converteddf = mfn.Convert(sourcedf)
        # Read some other mappings out of the conversion
        self.coltyped = mfn.coltyped
        self.coltypes = mfn.coltypes
        self.featuremapd = mfn.featuremapd   
        self.featuremaps = mfn.featuremaps  
        self.colmaps2d = mfn.colmaps2d
        
        self.numpydf = self.converteddf.to_numpy()
        self.numrow = self.numpydf.shape[0]
        self.numcold = self.numpydf.shape[1]
        # Precreate a large number of xgboost models : 
        #  -  firstly we create multiple models so we can push slightly different data at them to get a sense of the 
        #    confidence of prediction by looking at the different results.
        #  - secondly, we need a separate model to predict each column.
        
        self.models = dict()
        for cold in range(self.numcold):  
            coldname = self.converteddf.columns[cold]
            if self.coltyped[coldname] == 'raw':
                print('creating regressor')
                self.models[cold] = [XGBRegressor(verbosity=0, nthread=4) for j in range(self.models_for_confidence)]
            
            else:
                print('creating classifier')
                self.models[cold] = [XGBClassifier(verbosity=0, nthread=4) for j in range(self.models_for_confidence)]
 
        # Train multiple times on different subsets of the data to help us get a confidence interval. 
        for modelconf in range(self.models_for_confidence):

            # We create one model for every column.
            for cold in range(self.numcold):
                            
                # The x is all the columns except the y column we are training on.
                xcols = self.numpydf
                xtrain = np.delete(xcols, cold, 1)
                ytrain = self.numpydf[:,cold]
     
                # Train on a different subset of the data each time to add some randomness.
                xtrain, xtest, ytrain, ytest = model_selection.train_test_split(xtrain, ytrain, train_size=0.8)
                
                # print(xtrain)
                # print(ytrain)
                self.models[cold][modelconf].fit(xtrain, ytrain)

        
    def Predict(self, sourcedf):

        # Train, if we haven't already.
        if self.models is None:
            self.Train(sourcedf)
        
        self.predictedmeans = np.zeros((self.numrow, self.numcold))
        self.predictedstds  = np.zeros((self.numrow, self.numcold))
        
        # We create one model prediction for every column.
        for cold in range(self.numcold):  
            coldname = self.converteddf.columns[cold]
        
            # The x is all the columns except the y column we are predicting.
            xcols = self.numpydf
            xtest = np.delete(xcols, cold, 1)
        
            ytest = np.zeros((self.numrow, self.models_for_confidence))

            # Get multiple predictions back on subtly different training data to give us a variation of results and a confidence interval.
            # We don't accumulate predictions for the entire grid multiple times, because it might take a lot of memory to store.
            for modelconf in range(self.models_for_confidence):                
                ytest[:,modelconf] = self.models[cold][modelconf].predict(xtest)                
            
            # Accumulate means and standard deviations of the predictions per column.  Then we throw the detailed prediction data away.
            (self.predictedmeans[:,cold], self.predictedstds[:,cold]) = self.CalcMeanAndDeviation(ytest, self.coltyped[coldname])
        
        # Convert to dataframes with headers, easier for subsequent processing
        self.predictedmeans = pd.DataFrame(data=self.predictedmeans, columns=self.converteddf.columns)
        self.predictedstds  = pd.DataFrame(data=self.predictedstds,  columns=self.converteddf.columns)
        return(self.predictedmeans, self.predictedstds)       
            
    
    def SpotErrors(self, sourcedf):
    
        # Predict, if we haven't already.
        if self.predictedmeans is None:
            self.Predict(sourcedf)
    
        # Initially, we found no errors.
        self.boolerrors = pd.DataFrame(False, index=np.arange(len(self.converteddf.index)), columns=self.converteddf.columns)
        
        for coldname in self.converteddf.columns:

            for row in range(self.numrow):

                cellmean = self.predictedmeans[coldname][row]
                cellstd  = self.predictedstds[coldname][row]
   
                if self.coltyped[coldname] == 'labelencoded':
                    # If prediction <> actual and we are confident about the prediction
                    if cellmean != self.converteddf[coldname][row] and cellstd <= 0.3:
                        self.boolerrors[coldname][row] = True
                elif self.coltyped[coldname] == 'onehot':
                    if round(cellmean) != self.converteddf[coldname][row]:
                        self.boolerrors[coldname][row] = True                
                else:
                    # 100% confident prediction?
                    if cellstd == 0.0:
                        if cellmean != self.converteddf[coldname][row]:
                            self.boolerrors[coldname][row] = True
                    else:
                        # Not 100% confident prediction, use the zscore to decide if it's an error.
                        zscore = np.abs(cellmean - self.converteddf[coldname][row]) / cellstd       
                        # If bad, flag it as bad
                        if zscore >= self.zscore_for_error:
                            self.boolerrors[coldname][row] = True
                    
        return self.boolerrors
        
        
    def PrintErrors(self, sourcedf):
        # Spot errors, if we haven't already.
        if self.boolerrors is None:
            self.SpotErrors(sourcedf)
        
        for colsname in self.sourcedf.columns:
       
            for row in range(self.numrow):
                predicted = None
                
                # If it's a literal error, print it.
                if self.coltypes[colsname] == 'raw':
                    coldname = colsname
                    if self.boolerrors[coldname][row]: 
                        predicted = str(self.predictedmeans[coldname][row])
                        
                elif self.coltypes[colsname] == 'labelencoded':
                    coldname = colsname                
                    if self.boolerrors[coldname][row]:                    
                    
                        # What is the predicted value?  If std < 0.3 we are close to a single prediction
                        if self.predictedstds[coldname][row] < 0.3:
                            predicted = str(self.featuremapd[coldname][round(self.predictedmeans[coldname][row])])
                        else:
                            predicted = '(various)'
                        
                elif self.coltypes[colsname] == 'onehot':
                    # We have to look through all the destination columns, find what this mapped to (could be multiple), and then see
                    # whether we think that's an error.
                   
                    # Which cell did we originally think it is?  Unless we think that's an error, no need to go further.
                    coldname = self.colmaps2d[colsname][self.featuremaps[colsname][self.sourcedf[colsname][row]]]
                    
                    if self.boolerrors[coldname][row]:                    
                        predicted=[]
                        for coldname in self.colmaps2d[colsname]:
                        
                            cellmean = self.predictedmeans[coldname][row]
                            cellstd  = self.predictedstds[coldname][row]
                            if round(cellmean) == 1.0:
                                predicted.append(str(self.featuremapd[coldname]))
                        
                        if len(predicted) > 0:
                            if len(predicted) > 1:
                                predicted = '(any one of ' + ','.join(map(str,predicted)) + ')'
                            else:
                                predicted = str(predicted[0])
                        else:
                            predicted = '(no clear prediction)'

                if predicted is not None:
                    print('row ' + str(row) + ' column ' + colsname  + 
                    ': actual=' + str(self.sourcedf[colsname][row]) +
                    ' predicted=' + predicted)               


                    
                                 
        
    def CalcMeanAndDeviation(self, ypredictions, coltype):
                
        # If the column is a labelencoded, just calculate the standard deviation of boolean difference from the most common value
        if coltype == 'labelencoded':
            # Calculate the most common value
            if isinstance(ypredictions, list) or len(ypredictions.shape) == 1:
                mean = scipy.stats.mode(ypredictions).mode
                # And calculate the variation as mean deviation away from that, in boolean terms (different = 1, same = 0)
                std = np.mean(ypredictions != mean)
            else:
                assert(False, "weird")
                mean = scipy.stats.mode(ypredictions, axis=1).mode
                std = np.mean(ypredictions != mean, axis=1)
                mean = mean.reshape(-1)
        else:
            # Otherwise, conventional mean and standard deviation.
            if isinstance(ypredictions, list) or len(ypredictions.shape) == 1:
                mean = np.mean(ypredictions)           
                std  = np.std(ypredictions)
            else:
                mean = np.mean(ypredictions, axis=1)           
                std  = np.std(ypredictions, axis=1)

        return (mean, std)
        