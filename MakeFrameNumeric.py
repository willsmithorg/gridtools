import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


class MakeFrameNumeric:

    def __init__(self):

        self.maximum_cardinality_for_one_hot_encode = 10
    
        # The converted dataframe
        self.converteddf = None
        # Mapping from destination column to source column
        self.colmapd2s = None 
        # Mapping from source column to destination column
        self.colmaps2d = None         
        # Feature or features the column corresponds to : destination column
        self.featuremapd = None
        # Feature or features the column corresponds to : source column
        self.featuremaps = None
        # Type of destination column : raw, labelencoded, onehot        
        self.coltyped = None
        # Type of source column : raw, labelencoded, onehot
        self.coltypes = None        
        

    # xgboost needs the entire frame to be numeric.
    # So we'll convert strings to either one-hot or (if the cardinality is excessive for 1-hot) into a numerical label per unique string.
    def Convert(self, inputdf):
         
        if( not isinstance(inputdf, pd.DataFrame)):
            raise(TypeError,'inputdf must be a DataFrame not a ' + str(type(inputdf)))

        self.inputdf = inputdf    
        self.converteddf = pd.DataFrame()
        self.colmapd2s = dict()  
        self.colmaps2d = dict()           
        self.featuremapd = dict()
        self.featuremaps = dict()
        self.coltyped = dict()       
        self.coltypes = dict()
        
        #print('Converting...')
        for col in self.inputdf.columns:
            #print('Column : ' + col)
            #print('Type: ' + str(type(self.inputdf[col][0])))            
            cardinality = len(self.inputdf[col].unique())
            #print('Cardinality: ' + str(cardinality))
            
            # If string, one-hot (if not too many unique values) or feature encode.
            # TODO what about integers?  Some people use those for categories.  Maybe we can check the cardinality.
            if isinstance(self.inputdf[col][0], str):
                    
                    # Feature
                    label_encoder = LabelEncoder()
                    feature = label_encoder.fit_transform(self.inputdf[col])
                    feature = feature.astype(int)
                                    
                    if cardinality > self.maximum_cardinality_for_one_hot_encode:
                        self.converteddf[col] = feature
                        self.colmapd2s[col] = col
                        self.colmaps2d[col] = [col]
                        self.featuremapd[col] = dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))
                        self.featuremaps[col] = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_)))) 
                        self.coltyped[col] = 'labelencoded'
                        self.coltypes[col] = 'labelencoded'

                    else:
                        # One-hot
                        feature = feature.reshape(self.inputdf.shape[0], 1)
                        onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
                        feature = onehot_encoder.fit_transform(feature)
                        feature = feature.astype(int)
                       
                        self.colmaps2d[col] = []
                        self.coltypes[col] = 'onehot'
                        
                        self.featuremaps[col] = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_)))) 
                        
                        for f in range(feature.shape[1]):
                            convertedcol = col + '_' + str(f)
                            self.converteddf[convertedcol] = feature[:,f]
                            self.featuremapd[convertedcol] = label_encoder.classes_[f]
                            self.colmapd2s[convertedcol] = col
                            self.colmaps2d[col].append(convertedcol)                         
                            self.coltyped[convertedcol] = 'onehot'

            else:
                # Already numeric, copy directly.
                self.converteddf[col] = self.inputdf[col]
                self.colmapd2s[col] = col                
                self.colmaps2d[col] = [col]
                self.featuremapd[col] = None                
                self.featuremaps[col] = None
                self.coltyped[col] = 'raw'
                self.coltypes[col] = 'raw'                
            
        return self.converteddf
        