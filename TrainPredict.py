import pandas as pd
import numpy as np
import sklearn.model_selection as model_selection
from xgboost import XGBClassifier, DMatrix


# Only works for 2 columns so far, 1 is a kind of category and 2 is the data.

class TrainPredict:

    def __init__(self, df):
         
        if( not isinstance(df, pd.DataFrame)):
            raise(TypeError,'df must be a DataFrame not a ' + str(type(df)))

        self.__df = df
        self.__numpydf = self.__df.to_numpy()
        self.__numrows = self.__numpydf.shape[0]
        self.__numcols = self.__numpydf.shape[1]
 
        
        self.__models_for_confidence = 10
        self.__zscore_for_error = 5
        
        self.__models = [[XGBClassifier(verbosity=0) for j in range(self.__models_for_confidence)] for i in range(self.__numcols)]

 
    def Train(self):
        # Eventually we need to train for each row of the source dataframe.   For now, we're only going to train on the 3nd column.
        
        
        print(str(type(self.__models)))
        
        for modelconf in range(self.__models_for_confidence):

            # We create one model for every column.
            for col in range(self.__numcols):
                            
                # The x is all the columns except the y column we are training on.
                xcols = self.__numpydf
                xtrain = np.delete(xcols, col, 1)
                ytrain = self.__numpydf[:,col]
     
                # Train on a different subset of the data each time to add some randomness.
                xtrain, xtest, ytrain, ytest = model_selection.train_test_split(xtrain, ytrain, train_size=0.7)
                
                self.__models[col][modelconf].fit(xtrain, ytrain)
        
    def Predict(self):
        

        # We create one model prediction for every column.
        for col in range(self.__numcols):  
            
        
            # The x is all the columns except the y column we are predicting.
            xcols = self.__numpydf
            xtest = np.delete(xcols, col, 1)
        
            ytest = np.zeros((self.__numrows, self.__models_for_confidence))

            # Get multiple predictions back on subtly different training data to give us a variation of results and a confidence interval.
            for modelconf in range(self.__models_for_confidence):                
                ytest[:,modelconf] = self.__models[col][modelconf].predict(xtest)                
                
            for row in range(self.__numrows):
                cellmean = np.mean(ytest[row,:])
                cellstd  = np.std(ytest[row,:])
                
                if cellstd > 0:
                    zscore = np.abs(cellmean - self.__numpydf[row,col]) / cellstd
                else:
                    if cellmean != self.__numpydf[row,col]:
                        zscore = self.__zscore_for_error
                    else:
                        zscore = 0
                        
                
                if zscore >= self.__zscore_for_error:
                    diffmsg = '  <---- DIFF' 
                else: 
                    diffmsg = ''
                    
                print('['+str(row)+','+str(col)+'] actual:' + str(self.__numpydf[row,col]) + ' prediction:' + str(cellmean) + ' (stdev '+str(cellstd) + ')' + diffmsg)
                
            
        