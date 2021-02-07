import pandas as pd
import numpy as np
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
        self.__models = []
        
    def Train(self):
        # Eventually we need to train for each row of the source dataframe.   For now, we're only going to train on the 3nd column.
        
        # We create one model for every column.
        for col in range(self.__numcols):
            
            self.__models.append(XGBClassifier())
            
            # The x is all the columns except the y column we are training on.
            xcols = self.__numpydf
            xtrain = np.delete(xcols, col, 1)
            ytrain = self.__numpydf[:,col]
 
            print(xtrain.shape)   
            print(ytrain.shape)  
            
            self.__models[col].fit(xtrain, ytrain)
    
    def Predict(self):
        
        # We create one model prediction for every column.
        for col in range(self.__numcols):  
            
            # The x is all the columns except the y column we are predicting.
            xcols = self.__numpydf
            xtest = np.delete(xcols, col, 1)
            
            ytest = self.__models[col].predict(xtest)  
            for row in range(self.__numrows):
                print('['+str(row)+','+str(col)+'] actual:' + str(self.__numpydf[row,col]) + ' prediction:' + str(ytest[row]) + ('   <--- DIFF' if self.__numpydf[row,col]!=ytest[row] else ''))
                
            
        