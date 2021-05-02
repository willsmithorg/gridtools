import pandas as pd
import numpy as np
from Column import Column
import logging
from ColumnDeriver.Base import ColumnDeriverBase
import tensorflow_hub as hub
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')


class ColumnDeriverSentenceEmbedder(ColumnDeriverBase):

    description = "SentenceEmbedder of "
    
    # Doesn't make sense to apply this to itself.
    allowrecursive = False
    
    # It may produce 1:1 columns with the existing string, but we still like it, because it tells us 
    # things we didn't know about the string.
    allowOneToOne = True

    
    # Let's only do this on absolutely top level columns, because it's expensive (cpu, creates lots more columns).
    maxdepth = 0
    maybederived = False
    
    embedder_tensorflow_hub_url = "https://tfhub.dev/google/universal-sentence-encoder/4"

    def __init__(self):
        # This is slow because it has to cache the large model (or download it the first time) so do it just once.
        self.embedder = hub.load(self.embedder_tensorflow_hub_url)
        
        
    def IsApplicable(self, column):
        return column.dtype == 'object' and column.Some(column.StrMatches(' ')) and column.Most(column.StrMatches('[a-zA-Z]'))
        
    def Apply(self, column):
        embeddings = self.embedder(column.series)
        
        # We want to convert the tensorflow tensor that returns to a dict.
        # We do it like this  tensor => numpy array => dataframe (add column names) => dict.
        
        embeddings = embeddings.numpy()
        df = pd.DataFrame(embeddings, columns = [self.name+'_'+str(i+1)         for i in range(embeddings.shape[1])])
        embeddings_dict = df.to_dict('series')        
        
        return embeddings_dict
        



        
 

    
	