import pandas as pd
import scipy as scipy
import random
import numpy as np
import sklearn.model_selection as model_selection
import logging
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')

from xgboost import XGBClassifier, XGBRegressor, DMatrix
from MakeFrameNumeric import MakeFrameNumeric

from sklearn.preprocessing import LabelEncoder


# Train and predict a grid of tabular data (either a dict or a dataframe)
# 
# Nomenclature:
#     cold = column in the destination dataframe (labelencoded, onehotencoded)
#     cols = column in the source dataframe      (as passed in by the user)

class TrainPredict:

    def __init__(self):
         

        # Source dataframe, what the user passed in
        self.sourcedf = None
        # Converted dataframe, label encoded and 1-hot encoded
        self.converteddf = None
        # Numpy equivalent of converted dataframe, ready for XGBoost
        self.numpydf = None
        
        # Precalculate for ease of reference
        self.numrow_train = 0
        self.numrow_predict = 0
        self.numcold = 0  # Number of destination i.e. converted columns.

        
        # Constants
        self.models_for_confidence = 10
        self.train_data_subset = 0.8
        self.xgboost_subsample = 0.8
        self.numthreads_xgboost = 8
        self.xgboost_tree_method='auto' # gpu_hist = use gpu.   auto = default.

        # Store all the computed models
        self.models = None

        # Which columns we trained on.  This is a dictionary per destination predicted column.
        self.learned_cols = dict()
        
        
        # Make the predictions deterministic, so unit tests either succeed or fail and don't randomly change each time we run them.
        random.seed(42)
        np.random.seed(42)
        
        # We can explicitly only predict one row
        self.singlerowid = None
 
    def Train(self, sourcedf):
    
        if (isinstance(sourcedf, dict)):
            sourcedf = pd.DataFrame(sourcedf)
            
        if( not isinstance(sourcedf, pd.DataFrame)):
            raise(TypeError,'df must be a DataFrame not a ' + str(type(sourcedf)))

        if sourcedf is self.sourcedf:
            # We trained on this already.  Return early.
            # Caveat - maybe somebody changed the meta parameters above.  They might expect us to retrain.
            return
                    
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
        self.numrow_train = self.numpydf.shape[0]
        self.numcold = self.numpydf.shape[1]
        # Precreate a large number of xgboost models : 
        #  -  firstly we create multiple models so we can push slightly different data at them to get a sense of the 
        #    confidence of prediction by looking at the different results.
        #  - secondly, we need a separate model to predict each column.
        
        self.models = dict()
        for cold in range(self.numcold):  
            coldname = self.converteddf.columns[cold]

            if self.coltyped[coldname] == 'raw':
                #logging.debug('column ' + coldname + ': created regressor')
                self.models[coldname] = [XGBRegressor(                          tree_method=self.xgboost_tree_method, subsample=self.xgboost_subsample, verbosity=0, nthread=self.numthreads_xgboost, objective='reg:squarederror') for j in range(self.models_for_confidence)]
            elif self.coltyped[coldname] == 'onehot':
                #logging.debug('column ' + coldname + ': created binary classifier')            
                self.models[coldname] = [XGBClassifier(tree_method=self.xgboost_tree_method, subsample=self.xgboost_subsample, verbosity=0, nthread=self.numthreads_xgboost, objective='binary:logistic') for j in range(self.models_for_confidence)]            
            elif self.coltyped[coldname] == 'labelencoded':
                #logging.debug('column ' + coldname + ': created multiclass classifier')
                self.models[coldname] = [XGBClassifier(tree_method=self.xgboost_tree_method, subsample=self.xgboost_subsample, verbosity=0, nthread=self.numthreads_xgboost, objective='reg:logistic') for j in range(self.models_for_confidence)]
            else:
                raise(ValueError, 'Unrecognised coltyped : ' + self.coltyped[coldname])
                
            logging.debug('building model for column ' + coldname + ' type ' + self.coltyped[coldname])

            # The x is all the columns except the y column we are training on.

            x_all_cols = self.numpydf
            xtrain = self.__remove_predicted_columns_from_x(x_all_cols, coldname)                    
            ytrain = self.numpydf[:,cold]

            
             # Train multiple times on different subsets of the data to help us get a confidence interval. 
            for modelconf in range(self.models_for_confidence):
             
                # Train on a different subset of the data each time to add some randomness.
                if self.train_data_subset < 1.0:
                    xtrain_sub, _, ytrain_sub, _ = model_selection.train_test_split(xtrain, ytrain, train_size=self.train_data_subset)                
                else:
                    xtrain_sub = xtrain
                    ytrain_sub = ytrain

                   
                logging.debug(xtrain_sub.dtype)
                logging.debug(ytrain_sub.dtype)
                self.models[coldname][modelconf].fit(xtrain_sub, ytrain_sub)

                
    def __remove_predicted_columns_from_x(self, x_all_cols, coldname):
        # If we are training on a one-hot column, we have to delete all the other grouped one-hot, because these all came from
        # the same source column.
        if self.coltyped[coldname] == 'raw' or self.coltyped[coldname] == 'labelencoded':   

            coldid_to_remove = _column_index(self.converteddf, coldname)        

            x = np.delete(x_all_cols, coldid_to_remove, 1)
        elif self.coltyped[coldname] == 'onehot':
            colsname = self.colmapd2s[coldname]
            coldnames_to_remove = self.colmaps2d[colsname]

            coldids_to_remove = _column_index(self.converteddf, coldnames_to_remove)

            #logging.debug("Before deletion:")
            #logging.debug(coldid_to_remove)
            #logging.debug(x_all_cols)
            x = np.delete(x_all_cols, coldids_to_remove, 1)
            #logging.debug("After deletion:")
            #logging.debug(xtrain)
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


    # Predict just 1 column in the table, and optionally just a single row.
    def Predict(self, sourcedf, colsname, singlerowid = None):

        # Train.  It will return quickly if we already trained on this data.
        self.Train(sourcedf)
 
        if colsname not in self.sourcedf.columns:

            raise ValueError('Source column ' + colsname + ' not in source column list' + str(self.sourcedf.columns))
            
        # Are we looking for errors in an array of rows the same size as the training data, or just one row?
        self.singlerowid = singlerowid
        if self.singlerowid is None:

            self.numrow_predict = self.numrow_train
        else:
            if singlerowid > self.numrow_train:
                raise ValueError('Cannot train on row ' + str(singlerowid) + ' because the training data only had ' + str(self.numrow_train) + ' rows')

            self.numrow_predict = 1            
         
        # We create one model prediction for every destination column mapped from the source column.

        ytest = np.zeros((len(self.colmaps2d[colsname]), self.models_for_confidence, self.numrow_predict))

        
        for cold in range(len(self.colmaps2d[colsname])): 
            coldname = self.colmaps2d[colsname][cold]
            # The x is all the columns except the y column we are predicting.

            x_all_cols = self.numpydf            
            xtest = self.__remove_predicted_columns_from_x(x_all_cols, coldname) 
            # Save the list of columns we actually learned on.  This is useful in Explain, to understand the list of feature importances.
            self.learned_cols[coldname] = self.__remove_predicted_column_names(self.converteddf.columns, coldname)

            
            # If we are predicting just a single row, cut it out here.
            if self.singlerowid is not None:
                # Weird indexing just cuts one row out and makes sure the dimensions of the numpy array becomes 1 * num columns.  TODO is there a nicer syntax?
                xtest = xtest[self.singlerowid:self.singlerowid+1,:]        

            # For this single column, get multiple predictions back on subtly different training data to give us a variation of results and a confidence interval.
            # We don't accumulate predictions for the entire grid multiple times, because it might take a lot of memory to store.
            for modelconf in range(self.models_for_confidence):   
                ytest[cold,modelconf,:] = self.models[coldname][modelconf].predict(xtest)                
        
        # A 3-d array of predictions
        return ytest
          
        
# Helper function to do df.columns.get_loc(colnames) for a list of colnames in one go.
def _column_index(df, query_cols):
    cols = df.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols,query_cols,sorter=sidx)]
    
