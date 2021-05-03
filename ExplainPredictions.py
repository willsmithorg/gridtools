import pandas as pd
import numpy as np
import logging
import warnings

from InterpretPredictions import InterpretPredictions

logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')


class ExplainPredictions:
   
    # Load in all the predictions for a dataframe and which we think are wrong.
    # We will then call 'ExplainSinglePrediction' one by one on them.
    
    def __init__(self, trainPredictSelf, dfobserved, dfpredicted):
    
        self.dfobserved  = dfobserved
        self.dfpredicted = dfpredicted
        self.tps = trainPredictSelf
        
    
    def ExplainOneDifference(self, colname, rownum):
    
        results_labels, results_proba, importances = self.tps.TrainPredictSingleCell(colname, rownum)
        

        # TODO handle the case where the predictor 'changed it's mind' between the cross-validated
        # prediction passed in, and the single row prediction we just made.
        
        print('We think column', colname, 'row', rownum, 'is wrong')
        print('Observed : ', self.dfobserved[colname][rownum])
        print('We think the most likely are:')
        
        # Produce a list of tuples of form [ (prediction, probability) ]
        predictions = list(zip(results_labels, results_proba[0]))
        # Sort by probability descending.
        predictions.sort(key = lambda x:x[1], reverse = True)
        #print(predictions)
        
        # Print the first one.
        print(predictions[0][0], " : ", predictions[0][1] * 100, '%')
        
        # Continue printing any more that we think are more than 3% likely.
        i = 1
        while i < len(predictions):
            if predictions[i][1] >= 0.04:
                print(predictions[i][0], " : ", predictions[i][1] * 100, '%') 
                i += 1
            else:
                break
            

        # Now sort the feature importances by utility and take the top 1.
        importances.sort(key = lambda x:x[1], reverse = True)
        mostimportantcol = importances[0][0]
        
        # For this row, get the observed value in mostimportantcol, 
        # the describe the distribution of data in colname.  This shuold help to 
        # explain why the observed value is unlikely.
        observedsource = self.dfobserved[mostimportantcol][rownum]
        #print(observedsource)
        
        # Now get the list of target values in rownum (the row we think is incorrect) when mostimportantcol == observedsource
        
        target = self.dfobserved[self.dfobserved[mostimportantcol].eq(observedsource)]
        target = target[colname]
        targetproportions = target.value_counts(normalize=True)
        #print(targetproportions)
        
        # Print out the top few.
        print('Reason1:')
        print('\tWhen', mostimportantcol, '=',observedsource,', like this row')
        print('\tThen:')
        
        print('\t\t',colname, 'is', targetproportions.index[0], targetproportions[targetproportions.index[0]] * 100, '% of the time')
        
        i=1
        while i < len(targetproportions):
            if targetproportions[targetproportions.index[i]] >= 0.10:
                print('\t\t',colname, 'is', targetproportions.index[i], targetproportions[targetproportions.index[i]] * 100, '% of the time')
                i += 1
            else:
                break        
                
                
        # Try another reason based on the top 2 important columns.
        mostimportantcol1 = importances[0][0]
        mostimportantcol2 = importances[1][0]        
        observedsource1 = self.dfobserved[mostimportantcol1][rownum]
        observedsource2 = self.dfobserved[mostimportantcol2][rownum]
        target1 = self.dfobserved[self.dfobserved[mostimportantcol1].eq(observedsource1)]
        target2 = target1        [target1        [mostimportantcol2].eq(observedsource2)]
        target2 = target2[colname]
        targetproportions = target2.value_counts(normalize=True)
        # Print out the top few.
        print('Reason2:')
        print('\tWhen', mostimportantcol1, '=',observedsource1,', and', mostimportantcol2, '=', observedsource2, ',like this row')
        print('\tThen:')
        
        print('\t\t',colname, 'is', targetproportions.index[0], targetproportions[targetproportions.index[0]] * 100, '% of the time')
        
        i=1
        while i < len(targetproportions):
            if targetproportions[targetproportions.index[i]] >= 0.10:
                print('\t\t',colname, 'is', targetproportions.index[i], targetproportions[targetproportions.index[i]] * 100, '% of the time')
                i += 1
            else:
                break        
        