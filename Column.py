import pandas as pd
import scipy as scipy
import numpy as np
from cached_property import cached_property
import logging
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')


class Column:

    def __init__(self, series):
    
        self.series = series
        self.parent = None        
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
    @property
    def nunique(self):
        return self.series.nunique()
    
    def MakeChild(self, s):
        self.children.append(s)
        s.parent = self
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
               ).format(
                        name=  self.name,
                        size=  self.size,
                        dtype= self.dtype,
                        depth= self.depth,
                        children=','.join(self.ChildNames()),
                        parent='-' if self.parent == None else self.parent.name
                        )

        datastr = str(self.series.head())
        
        return colstr + datastr
        
    
     
            
    
