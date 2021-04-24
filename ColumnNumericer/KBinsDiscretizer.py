import pandas as pd
import numpy as np
from Column import Column
import logging
from ColumnNumericer.Base import ColumnNumericerBase
from sklearn.preprocessing import KBinsDiscretizer
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')


# This helps us to convert all our columns to discrete, so we can use a Classifier everywhere, not a regressor.
class ColumnNumericerKBinsDiscretizer(ColumnNumericerBase):

    # Don't create more than this many bins, even for large datasets.  We want to give the classifier a decent
    # chance of guessing the right bin.
    max_bins = 20
    samples_per_bin = 4
    min_bins_ideally = 5
    min_unique = 10
    
    def __init__(self):
        # We don't start up the discretizer until we know about the data, because it affects our choice of bins.
        self.discretizer = None

    def IsApplicable(self, column, target):
        target = target.upper()
        
        # We only want to onehot encode X values , not y, and ones with lots of unique values.
        return column.IsNumeric() and target == 'Y' and column.nunique >= self.min_unique
        
    def Apply(self, column):
        # Create the discretizer now.  The choice of bins depends on the data we observe.
        self.discretizer = KBinsDiscretizer(n_bins=self.ChooseBins(column), encode='ordinal', strategy='quantile') 
        
        inp = column.series.values
        inp = inp.reshape(-1,1)        
        feature = self.discretizer.fit_transform(inp)   
        # print('result of discretization:')
        feature = feature.reshape(-1)
        # print(feature)
        return feature
        
    # Inverse won't be perfect because we binned into discrete bins.
    def Inverse(self, numpy_array):
        numpy_array = numpy_array.reshape(-1,1)
        inverse = self.discretizer.inverse_transform(numpy_array)
        inverse = inverse.reshape(-1)
        return inverse


    # Choose enough bins to have several values per bin, ideally around 4.  But not so many bins that we have trouble predicting
    # accurately.
    def ChooseBins(self, column):
        bins = np.floor(column.nunique / self.samples_per_bin).astype('int')
        if bins > self.max_bins:
            bins = self.max_bins
        elif bins < self.min_bins_ideally:
            # 4 values per bin took us a bit low.  Try to go higher.
            bins = np.floor(column.nunique / 2).astype('int')
            
        if bins > self.max_bins:
            bins = self.max_bins
            
        return bins    
            
	