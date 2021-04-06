import pandas as pd
import numpy as np
from Column import Column
import logging
from ColumnNumericer.Base import ColumnNumericerBase
from sklearn.preprocessing import LabelEncoder
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')


class ColumnNumericerLabelEncoded(ColumnNumericerBase):

    def __init__(self):
        self.label_encoder = LabelEncoder()

    def IsApplicable(self, column, target):
        # We can label encode any field that's not already numeric, regardless of whether it's X or Y.
        return not column.IsNumeric()
        
    def Apply(self, column):
    
        feature = self.label_encoder.fit_transform(column.series.values)
        feature = feature.astype(int)                    
        return feature
        
    
	