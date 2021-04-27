import pandas as pd
import numpy as np
import warnings
import logging
import random
from sklearn.datasets import make_classification
from sklearn.preprocessing import KBinsDiscretizer
from MakeNumericColumns import MakeNumericColumns
from TrainPredictSelf import TrainPredictSelf


logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')



class SyntheticError:
 
 
    def __init__(self):
    
        pass
        
        
    def MakeClassification(self, n_samples = 10, 
                                 n_features = 6,
                                 n_informative = 3, 
                                 n_redundant = 3, 
                                 n_classes = 5,                             
                                 n_clusters_per_class = 1, 
                                 n_bins=5,
                                 n_discretized = 4):
    
        x, y = make_classification(n_samples = n_samples, 
                                   n_features = n_features, 
                                   n_informative = n_informative,
                                   n_redundant = n_redundant, 
                                   n_classes = n_classes, 
                                   n_clusters_per_class = n_clusters_per_class)
                                   
        df = pd.DataFrame()
        
        #print(x)
        # Randomly shuffle the columns, because make_classification puts the informative columns first.
        np.random.shuffle(np.transpose(x))


        num_discretized = 0
        for i in range(x.shape[1]):
            # Convert some of the floating point data into categorized.
            if (num_discretized < n_discretized):                            
                col =  KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile').fit_transform(x[:,i].reshape(-1,1)).astype(int).reshape(-1)
                num_discretized += 1
            else:
                col = x[:,i]                    
            df['col'+str(i)] = col
            
        df['col'+str(x.shape[1])] = y
        #print(df)        

        return df
        
    def HowPredictable(self, df):
    
        tps = TrainPredictSelf()
        
        results, proba = tps.Train(df)
        results, _ = tps.SinglePredictionPerCell(results, proba)
        boolSame = results.eq(df)
        
        # Prints the sum of all the Trues (where we correctly predicted the input dataframe) / the total size.
        # This is a single float, and tells us how 'predictable' the entire dataframe is.
        
        return (boolSame.sum().sum() / boolSame.size)
        
    # Introduce 'quantity' errors to an existing dataframe.  Return the new dataframe and a boolean array of the same shape, 
    # listing where the errors were introduced.
    def CreateErrors(self, df, quantity=1):
    
        # Create an identically sized array to tell us which cells we changed.
        boolChanged = df.copy()
        for col in boolChanged.columns:            
            boolChanged[col].values[:] = False
            boolChanged[col] = boolChanged[col].astype(bool)
            
        for i in range(quantity):
        
            # Choose what cell to change.  Make sure it's ot a cell we've already changed.
            different = False
            while not different:
                targetcolumn = random.randint(0, len(df.columns)-1)
                targetrow    = random.randint(0, len(df)-1)                
                if boolChanged.iloc[targetrow, targetcolumn] == False:
                    different = True
                
            
            column_to_perturb = df.columns[targetcolumn]
            # Discrete - make sure we change it to something different.
            if self.LooksLikeDiscreteColumn(df[column_to_perturb]):                
                different = False
                while not different:
                    new = random.randint(min(df[column_to_perturb]), max(df[column_to_perturb]))  
                    if new != df.iloc[targetrow, targetcolumn]:
                        different = True
            else:
            # Continuous - choose any random value within the existing range.
                new = random.uniform(min(df[column_to_perturb]), max(df[column_to_perturb]))
                        
            df.iloc[targetrow, targetcolumn] = new
            boolChanged.iloc[targetrow, targetcolumn] = True
            
        return df, boolChanged
            
    # It's a discrete column if it contains integers.
    def LooksLikeDiscreteColumn(self, series):

        if series.dtype == 'int64' or series.dtype == 'int32':
            return True
        else:
            return False