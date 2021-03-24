import pandas as pd
import numpy as np
from Column import Column
import logging

from nltk.tokenize import RegexpTokenizer

from ColumnDeriver.Base import ColumnDeriverBase
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')


class ColumnDeriverTokenizerCharDecimal(ColumnDeriverBase):

    name = "tokenizer_char_decimal"
    description = "Results of tokenizing the column into characters and decimal pieces "
    
    # Don't apply if any strings are longer than this.
    maxlen = 30
    
    # Doesn't make sense to apply this to itself.
    allowrecursive = False

    def IsApplicable(self, column):
        return column.dtype == 'object' and all(column.series.fillna('').map(len) < self.maxlen)
        
    def Apply(self, column):
        tokenizer_c = RegexpTokenizer(r'[a-zA-Z]+')
        ctok = [ tokenizer_c.tokenize(s) for s in column.series ]
        
        tokenizer_d = RegexpTokenizer('\d+')
        dtok = [ tokenizer_d.tokenize(s) for s in column.series ]
        
        tokenizer_digitfirst = RegexpTokenizer('^\d')
        dftok = [ tokenizer_digitfirst.tokenize(s) for s in column.series ]
        
         # A dataframe just of the character tokens.
        dfc = pd.DataFrame.from_records(ctok)
        dfc.columns = ['characters'+str(i+1) for i in range(len(dfc.columns))]
        # A dataframe just of the numeric tokens.
        dfd = pd.DataFrame.from_records(dtok)
        dfd.columns = ['digits'+str(i+1) for i in range(len(dfd.columns))]
        # A dataframe of booleans, specifying whether the string started with the digits (True)
        dfdf = pd.DataFrame.from_records(dftok)
        dfdf = dfdf.applymap(type).eq(str)
        dfdf.columns = ['digitsfirst']

        combined = pd.concat([dfc,dfd,dfdf], axis=1)
        
        return combined.to_dict('series')
	