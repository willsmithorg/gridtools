import pandas as pd
import scipy as scipy
import numpy as np
import sklearn.model_selection as model_selection
from xgboost import XGBClassifier, DMatrix
from MakeFrameNumeric import MakeFrameNumeric

# Only works for 2 columns so far, 1 is a kind of category and 2 is the data.

class TrainPredict:

    def __init__(self):
         

        self.sourcedf = None
        self.converteddf = None
        self.numpydf = None
        self.numrow = 0
        self.numcold = 0  # Number of destination i.e. converted columns.
        self.coltyped = None
        
        self.models_for_confidence = 10
        self.zscore_for_error = 3
        


        self.models = None
        self.predictedmeans = None
        self.predictedstds = None
        self.boolerrors = None
        
 
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
        
        self.numpydf = self.converteddf.to_numpy()
        self.numrow = self.numpydf.shape[0]
        self.numcold = self.numpydf.shape[1]
        self.colname2loc = dict(zip(self.converteddf.columns, range(self.numcold)))        
        # Precreate a large number of xgboost models : 
        #  -  firstly we create multiple models so we can push slightly different data at them to get a sense of the 
        #    confidence of prediction by looking at the different results.
        #  - secondly, we need a separate model to predict each column.
        
        self.models = [[XGBClassifier(verbosity=0, nthread=4) for j in range(self.models_for_confidence)] for i in range(self.numcold)]
 
        # Train multiple times on different subsets of the data to help us get a confidence interval. 
        for modelconf in range(self.models_for_confidence):

            # We create one model for every column.
            for cold in range(self.numcold):
                            
                # The x is all the columns except the y column we are training on.
                xcols = self.numpydf
                xtrain = np.delete(xcols, cold, 1)
                ytrain = self.numpydf[:,cold]
     
                # Train on a different subset of the data each time to add some randomness.
                xtrain, xtest, ytrain, ytest = model_selection.train_test_split(xtrain, ytrain, train_size=0.7)
                
                # print(xtrain)
                # print(ytrain)
                self.models[cold][modelconf].fit(xtrain, ytrain)

        
    def Predict(self):

        # Train, if we haven't already.
        if self.models is None:
            self.Train()
        
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
            
        return(self.predictedmeans, self.predictedstds)       
            
    
    def SpotErrors(self):
    
        # Predict, if we haven't already.
        if self.predictedmeans is None:
            self.Predict()
    
        # Initially, we found no errors.
        self.boolerrors = np.full((self.numrow, self.numcold), False)
        for cold in range(self.numcold):  
            coldname = self.converteddf.columns[cold]

            for row in range(self.numrow):

                cellmean = self.predictedmeans[row,cold]
                cellstd  = self.predictedstds[row,cold]
   
                #TODO work on the logic below
                if self.coltyped[coldname] == 'labelencoded':
                    if cellmean != self.numpydf[row,cold] and cellstd < 0.2:
                        zscore = self.zscore_for_error
                    else:
                        zscore = 0
                else:
                    if cellstd == 0:
                        if cellmean != self.numpydf[row,cold]:
                            zscore = self.zscore_for_error
                        else:
                            zscore = 0
                    else:
                        zscore = np.abs(cellmean - self.numpydf[row,cold]) / cellstd
                        
                # If bad, flag it as bad
                if zscore >= self.zscore_for_error:
                    self.boolerrors[row, cold] = True
                    
        return self.boolerrors
        
        
    def PrintErrors(self):
        # Spot errors, if we haven't already.
        if self.boolerrors is None:
            self.SpotErrors()
        
        for cols in range(len(self.sourcedf.columns)):
        
            colsname = self.sourcedf.columns[cols]
            for row in range(self.numrow):

                # If it's a literal error, print it.
                if self.coltypes[colsname] == 'raw':
                    cold = colsname
                    colnameloc = self.colname2loc[cold]
                    if self.boolerrors[row, colnameloc]:                    
                        print('Raw column ' + colsname + ' row ' + str(row) + 
                        ': actual=' + str(self.df[cold][row]) +
                        ' predicted' + str(self.predictedmeans[row, colnameloc]))
                    
                elif self.coltypes[colsname] == 'labelencoded':
                    cold = colsname                
                    colnameloc = self.colname2loc[colsname]
                    if self.boolerrors[row, colnameloc]:
                    
                        # What is the predicted value?  If std < 0.3 we are close to a single prediction
                        if self.predictedstds[row, colnameloc] < 1:
                            predicted = str(self.featuremapd[cold][round(self.predictedmeans[row, colnameloc])])
                        else:
                            predicted = '(various)'
                        
                        print('Raw column ' + colsname + ' row ' + str(row) + 
                        ': actual=' + self.sourcedf[colsname][row] +
                        ' predicted=' + predicted )           
        
    def CalcMeanAndDeviation(self, ypredictions, coltype):
                
        # If the column is a labelencoded, just calculate the standard deviation of boolean difference from the most common value
        if coltype == 'labelencoded':
            # Calculate the most common value
            mean = scipy.stats.mode(ypredictions).mode
            # And calculate the variation as mean deviation away from that.
            std = np.mean(ypredictions != mean)
        else:
            # Otherwise, conventional mean and standard deviation.
            mean = np.mean(ypredictions)           
            std  = np.std(ypredictions)
            
        return (mean, std)
        