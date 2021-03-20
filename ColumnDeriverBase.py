import pandas as pd
import numpy as np
import logging
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')


class ColumnDeriverBase:

    subclasses = []
    
    def __init_subclass__(self, **kwargs):
        super().__init_subclass__(**kwargs)
        self.subclasses.append(self)
        
    