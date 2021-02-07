import pandas as pd
import numpy as np
from xgboost import XGBClassifier, DMatrix


# Only works for 2 columns so far, 1 is a kind of category and 2 is the data.

class TrainPredict:

    def __init__(self, df):
         
        if( not isinstance(df, pd.DataFrame)):
            raise(TypeError,'df must be a DataFrame not a ' + str(type(df)))

        self.__df = df
        self.__npdf = self.__df.to_numpy()
 
    def Train(self):
        # Eventually we need to train for each row of the source dataframe.   For now, we're only going to train on the 3nd column.
        self.__model = XGBClassifier()
        
        print(self.__npdf[:,0].shape)
        
        self.__model.fit(self.__npdf[:,0:1], self.__npdf[:,2])
    
    def Predict(self):
                
        prediction = self.__model.predict(self.__npdf[:,0:1])  
        for i in range(self.__npdf.shape[0]):
            print('row:' + str(i) + ' actual:' + str(self.__npdf[i,2]) + ' prediction:' + str(prediction[i]))
            
            
        