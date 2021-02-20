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
        self.confidence_to_keep_column = 0.25
        self.std_for_single_prediction_labelencoded = 0.3
        self.train_data_subset = 0.8
        self.numthreads_xgboost = 4

        self.models = None
        self.predictedmeans = None
        self.predictedstds = None
        self.boolerrors = None
        
        # Make the predictions deterministic, so unit tests either succeed or fail and don't randomly change each time we run them.
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
        self.colmapd2s = mfn.colmapd2s
        
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
                print('column ' + coldname + ': created regressor')
                self.models[coldname] = [XGBRegressor(verbosity=0, nthread=self.numthreads_xgboost, objective='reg:squarederror') for j in range(self.models_for_confidence)]
            elif self.coltyped[coldname] == 'onehot':
                print('column ' + coldname + ': created binary classifier')            
                self.models[coldname] = [XGBClassifier(verbosity=0, nthread=self.numthreads_xgboost, objective='binary:logistic') for j in range(self.models_for_confidence)]            
            else:
                print('column ' + coldname + ': created multiclass classifier')
                self.models[coldname] = [XGBClassifier(verbosity=0, nthread=self.numthreads_xgboost, objective='reg:logistic') for j in range(self.models_for_confidence)]
 
        # Train multiple times on different subsets of the data to help us get a confidence interval. 
        for modelconf in range(self.models_for_confidence):

            # We create one model for every column.
            for cold in range(self.numcold):
                coldname = self.converteddf.columns[cold] 
                print('building model for column ' + coldname + ' type ' + self.coltyped[coldname])
                # The x is all the columns except the y column we are training on.
                x_all_cols = self.numpydf
                xtrain = self.__remove_predicted_columns_from_x(x_all_cols, coldname)                    
                ytrain = self.numpydf[:,cold]
     
                # Train on a different subset of the data each time to add some randomness.
                xtrain, _, ytrain, _ = model_selection.train_test_split(xtrain, ytrain, train_size=self.train_data_subset)
                
                # print(xtrain)
                # print(ytrain)
                self.models[coldname][modelconf].fit(xtrain, ytrain)

    def __remove_predicted_columns_from_x(self, x_all_cols, coldname):
        # If we are training on a one-hot column, we have to delete all the other grouped one-hot, because these all came from
        # the same source column.
        if self.coltyped[coldname] == 'raw' or self.coltyped[coldname] == 'labelencoded':   
            coldid_to_remove = column_index(self.converteddf, coldname)        
            x = np.delete(x_all_cols, coldid_to_remove, 1)
        elif self.coltyped[coldname] == 'onehot':
            colsname = self.colmapd2s[coldname]
            coldnames_to_remove = self.colmaps2d[colsname]
            coldids_to_remove = column_index(self.converteddf, coldnames_to_remove)
            #print("Before deletion:")
            #print(coldid_to_remove)
            #print(x_all_cols)
            x = np.delete(x_all_cols, coldids_to_remove, 1)
            #print("After deletion:")
            #print(xtrain)
        else:
            raise(TypeError,'coltyped must be one of (raw, labelencoded, onehot) not ' + self.coltyped)            
        return x    
        
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
            x_all_cols = self.numpydf            
            xtest = self.__remove_predicted_columns_from_x(x_all_cols, coldname)                            
            ytest = np.zeros((self.numrow, self.models_for_confidence))

            # Get multiple predictions back on subtly different training data to give us a variation of results and a confidence interval.
            # We don't accumulate predictions for the entire grid multiple times, because it might take a lot of memory to store.
            for modelconf in range(self.models_for_confidence):                
                ytest[:,modelconf] = self.models[coldname][modelconf].predict(xtest)                
            
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
                    if cellmean != self.converteddf[coldname][row] and cellstd <= self.std_for_single_prediction_labelencoded:
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
                stdev = None
                
                # If it's a literal error, print it.
                if self.coltypes[colsname] == 'raw':
                    coldname = colsname
                    if self.boolerrors[coldname][row]: 
                        predicted = str(self.predictedmeans[coldname][row])
                        stdev     = str(self.predictedstds[coldname][row])
                        
                elif self.coltypes[colsname] == 'labelencoded':
                    coldname = colsname                
                    if self.boolerrors[coldname][row]:                    
                    
                        # What is the predicted value?  If std < 0.3 we are close to a single prediction
                        if self.predictedstds[coldname][row] < self.std_for_single_prediction_labelencoded:
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
                    ' predicted=' + predicted + 
                    (' stdev='+ stdev if stdev is not None else ''))              


                    
                                 
        
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
            
            
        
# Helper function to do df.columns.get_loc(colnames) for a list of colnames in one go.
def column_index(df, query_cols):
    cols = df.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols,query_cols,sorter=sidx)]
    