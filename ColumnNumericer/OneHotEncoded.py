import pandas as pd
import numpy as np
from Column import Column
import logging
from ColumnNumericer.Base import ColumnNumericerBase
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')


class ColumnNumericerOneHotEncoded(ColumnNumericerBase):

    # Don't convert a categorical field with more than this amount of unique fields to one-hot
    max_nunique = 20
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.onehot_encoder = OneHotEncoder(sparse=False, categories='auto')

    def IsApplicable(self, column, target):
        # We only want to onehot encode X values , not y.
        return not column.IsNumeric() and target.upper() == 'X' and column.nunique <= self.max_nunique 
        
    def Apply(self, column):
    
        feature = self.label_encoder.fit_transform(column.series.values)
        feature = feature.reshape(column.series.values.shape[0], 1)
        feature = self.onehot_encoder.fit_transform(feature)
                
        return feature
        
    
	