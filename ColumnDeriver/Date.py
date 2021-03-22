import pandas as pd
import numpy as np
from Column import Column
import logging
from ColumnDeriver.Base import ColumnDeriverBase
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')


class ColumnDeriverDate(ColumnDeriverBase):

    name = "date"
    description = "Date parse of "
   
    # Doesn't make sense to apply this to itself.
    allowrecursive = False

    # If this many rows match a regex, the column is a date.    
    matchregexes =  [ '(?:19|20|21|22)\d\d\d\d\d\d'  ]
    captureregexes = [ '(?P<year>(?:19|20|21|22)\d\d)(?P<month>\d\d)(?P<day>\d\d)' ]
    
    def IsApplicable(self, column):
        return column.dtype == 'object'
        
    def Apply(self, column):
    
        for matchregex, captureregex in zip(self.matchregexes, self.captureregexes):
            matches_bool =  self.StrMatches(column, matchregex)        
            # If the number NOT matching is low, parse out a date (we do it this way because we might have NAs).
            
            if self.Some(column, self.StrMatches(column, matchregex)):
                # print(column.name, 'is a date using', matchregex)
            
                df = column.series.str.extract( pat = captureregex).astype(str)
                # print(df)
                
                return { self.name + '_year'  :  df.year,
                         self.name + '_month' :  df.month,
                         self.name + '_day'   :  df.day }

                     
            else:  
                # print(column.name, 'is not a date using', matchregex)
                pass
                        

        # If we get through to here, are there no dates and hence no derived columns.
        return { }
        
        

    
	