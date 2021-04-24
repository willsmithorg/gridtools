import pandas as pd
import numpy as np
import warnings
import logging
from sklearn.datasets import make_classification
from sklearn.preprocessing import KBinsDiscretizer
from MakeNumericColumns import MakeNumericColumns
from TrainPredictSelf import TrainPredictSelf

logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')



class SyntheticError:
 
 
    def __init__(self):
    
        pass
        
        
    def MakeClassification(self, n_samples = 100, 
                                 n_features = 60,
                                 n_informative = 30, 
                                 n_redundant = 30, 
                                 n_classes = 5,                             
                                 n_clusters_per_class = 1, 
                                 n_bins=5,
                                 n_discretized = 40):
    
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
        
