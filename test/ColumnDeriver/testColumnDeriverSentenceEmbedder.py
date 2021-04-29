import unittest
import pandas as pd
import numpy as np
from Column import Column
from AddDerivedColumns import AddDerivedColumns

import logging
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')

import os

# Make tensorflow startup much quieter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)


class TestColumnDeriverSentenceEmbedder(unittest.TestCase):

    def setUp(self):
        self.col1 = Column(pd.Series(['abc','def', 'ghij', 'abc'], name='col1'))
        self.col2 = Column(pd.Series([1, 2, 4],                    name='col2'))
        self.col3 = Column(pd.Series(['these are proper sentences.  they should be embedded',
                                      'and here''s another one'],                    name='col3'))
 
        self.adc = AddDerivedColumns()
        self.adc.Register('SentenceEmbedder') 
        self.expected_embedding_size = 512
        

    # We shouldn't get a derived column on a numeric 
    @unittest.skipIf(os.environ.get('SKIPSLOW'), 'skipping slow tests')
    def testSentenceEmbedderNotApplicable(self):
        newcols = self.adc.Process(self.col1)        
        self.assertEqual(len(newcols), 0)
        
        newcols = self.adc.Process(self.col2)        
        self.assertEqual(len(newcols), 0)

    # We should get a derived column on a string column.
    @unittest.skipIf(os.environ.get('SKIPSLOW'), 'skipping slow tests')
    def testSentenceEmbedderOnSentences(self):
        newcols = self.adc.Process(self.col3)    
        # 1 new columns got created from the embedding.
        self.assertEqual(len(newcols), self.expected_embedding_size)
        self.assertIsInstance(newcols[0], Column)
        self.assertEqual(newcols[0].name, 'col3.SentenceEmbedder_1') 

        # Check the embeddings looks sane.  Each value should be -1<value<1 and the mean should not be 0, to make sure [0,0,0,0,0] doesn't pass.
        self.assertEqual(len(newcols[0].series),2)  # We embedded 2 sentences.
        self.assertTrue(np.max([ x.series[0] for x in newcols]) < 1)
        self.assertTrue(np.min([ x.series[0] for x in newcols]) > -1)
        self.assertTrue(np.mean([ x.series[0] for x in newcols]) != 0)        

        
if __name__ == '__main__':
    # Disable GPU because it makes this test slower to run.
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    unittest.main()
    

