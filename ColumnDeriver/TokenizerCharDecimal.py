import pandas as pd
import numpy as np
from Column import Column
import logging

from nltk.tokenize import RegexpTokenizer

from ColumnDeriver.Base import ColumnDeriverBase
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')


# Converts strings like 'AA-123BB453C" to "AA", "123", "BB" "453", "C", in case these have separate meaning and help the learning algo.
# The numeric and character derived columns are numbered separately.
# We also capture whether the numerics came first, to differentiate between "123ABC" and "ABC123".

class ColumnDeriverTokenizerCharDecimal(ColumnDeriverBase):

    description = "Results of tokenizing the column into characters and decimal pieces "
    
    # Don't apply if any strings are longer than this.
    maxlen = 30
    
    # Doesn't make sense to apply this to itself.
    allowrecursive = False

    def IsApplicable(self, column):
        return column.dtype == 'object' and all(column.series.fillna('').map(len) < self.maxlen)
        
    def Apply(self, column):
    
        # A dataframe just of the character tokens.
    
        tokenizer_c = RegexpTokenizer(r'[a-zA-Z]+')
        ctok = [ tokenizer_c.tokenize(s) for s in column.series ]
        # We also produce the tokens in the reverse order in case 'last characters', '2nd last numeric' etc mean something.
        ctok_reverse = [elem[::-1] for elem in ctok]
        
        dfc =  pd.DataFrame.from_records(ctok)   
        dfcr = pd.DataFrame.from_records(ctok_reverse)         
        dfc.columns = [self.name+'_characters'+str(i+1)         for i in range(len(dfc.columns))]
        dfcr.columns = [self.name+'_reversecharacters'+str(i+1) for i in range(len(dfcr.columns))]
        
         # A dataframe just of the numeric tokens.      
        tokenizer_d = RegexpTokenizer(r'\d+')
        dtok = [ tokenizer_d.tokenize(s) for s in column.series ]
        # We also produce the tokens in the reverse order in case 'last characters', '2nd last numeric' etc mean something.        
        dtok_reverse = [elem[::-1] for elem in dtok]


        dfd = pd.DataFrame.from_records(dtok)
        dfdr = pd.DataFrame.from_records(dtok_reverse)         
        
        dfd.columns = [self.name+'_digits'+str(i+1)         for i in range(len(dfd.columns))]
        dfdr.columns = [self.name+'_reversedigits'+str(i+1) for i in range(len(dfdr.columns))]

         # A dataframe of booleans, specifying whether the string started with the digits (True)
       
        tokenizer_digitfirst = RegexpTokenizer(r'^\d')
        dftok = [ tokenizer_digitfirst.tokenize(s) for s in column.series ]
             
        dfdf = pd.DataFrame.from_records(dftok)
        dfdf = dfdf.applymap(type).eq(str)
        dfdf.columns = [self.name+'_digitsfirst']
        
        combined = pd.concat([dfc,dfcr, dfd, dfdr, dfdf], axis=1)
      
        return combined.to_dict('series')
	