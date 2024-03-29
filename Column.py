import pandas as pd
import scipy as scipy
import numpy as np
from cached_property import cached_property
import logging
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')


class Column:

    def __init__(self, series):
    
        if not isinstance(series, pd.Series):
            raise TypeError('input series must be a pandas Series, not a ' + str(type(series)))
    
        self.series = series
        self.parent = None 
        self.ancestor = None
        self.children = []
        self.depth = 0
        # Which deriver was used to create this column.
        self.deriver = None
    
    @property
    def size(self):
        return self.series.size

    @property
    def name(self):
        return self.series.name

    @name.setter
    def name(self, newname):
        self.series.name = newname
        
    @property
    def dtype(self):
        return self.series.dtype
        
    # Number of unique elements.  We expect this might be expensive to compute so we cache it.
    @cached_property
    def nunique(self):
        return self.series.nunique(dropna=False)
    
    # Make s a child column of this column.
    def MakeChild(self, s):
        self.children.append(s)
        s.parent = self
        
        # Also record the top level ancestor.  This makes it quick to remove all columns
        # that were derived from an ancestor, over several levels.
        if self.ancestor is None:
            s.ancestor = self
        else:
            s.ancestor = self.ancestor
            
        s.depth = self.depth+1
        
    
    def ChildNames(self):
        return [c.name for c in self.children]
        
    def __str__(self):
    
        colstr =('Name : {name}\n'
                'Size : {size}\n'
                'Dtype : {dtype}\n'
                'Depth : {depth}\n'
                'Children: [{children}]\n'
                'Parent: {parent}\n'
                'Ancestor: {ancestor}\n'
               ).format(
                        name=  self.name,
                        size=  self.size,
                        dtype= self.dtype,
                        depth= self.depth,
                        children=','.join(self.ChildNames()),
                        parent='-'   if self.parent   is None else self.parent.name,
                        ancestor='-' if self.ancestor is None else self.ancestor.name
                        
                        )

        datastr = str(self.series.head(10))
        
        return colstr + datastr
        
 
    # Operations on the underlying series.
    def StrContains(self, substring):
        return self.series.str.contains(substring, regex=False)
        
    def StrMatches(self, regex):
        return self.series.str.contains(regex, regex=True)

    def IsNumeric(self):
        return self.series.dtype == 'int64' or self.series.dtype == 'float64'
                       
    def Most(self, boolseries, threshold=0.8):
        if boolseries.sum() >= threshold * self.size:
            return True
        else:
            return False
 
    def All(self, boolseries, threshold=1.0):
        return self.Most(boolseries, threshold)
        
    def Some(self, boolseries, threshold=0.5):
        return self.Most(boolseries, threshold)          
        
    def AFew(self, boolseries, threshold=0.2):
        return self.Most(boolseries, threshold) 

    def NotAny(self, boolseries, threshold=0.0):
        return not self.Most(boolseries, threshold)
     
    def IsCategorical(self):    
        if self.dtype == 'object':
            return True
        elif self.dtype == 'int64':        
            return True
        elif self.dtype == 'int32':
            return True
        elif self.dtype == 'float64':           # Even floats should be true because they will have been 
                                                # through KBinsDiscretizer (for Y only)
            return True
        elif self.dtype == 'bool':
            return True
        else:
            raise RuntimeError('unknown IsCategorical for dtype = ', self.series.dtype)
        
    def IsBinary(self):
        return self.IsCategorical() and self.nunique == 2
    
    # Report if two series are 1:1.  In some cases, this will help avoid unnecessary deriving.
    # Based on https://stackoverflow.com/questions/50643386/easy-way-to-see-if-two-columns-are-one-to-one-in-pandas      
    def IsOneToOne(self, col2):
    
        # Two fast tests to prevent the effort of creating a dataframe lower down for a more
        # comprehensive test.
        if self.size != col2.size:
            return False
        
        if self.nunique != col2.nunique:
            return False
            
        df = pd.DataFrame()
        df['col1']=self.series
        df['col2']=col2.series
        first  = df.drop_duplicates(['col1', 'col2']).groupby('col1')['col2'].count().max()
        
        if first != 1:
            return False
            
        second = df.drop_duplicates(['col1', 'col2']).groupby('col2')['col1'].count().max()
        
        return second == 1
            