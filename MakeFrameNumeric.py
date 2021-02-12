import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


class MakeFrameNumeric:

    def __init__(self, df):
         
        if( not isinstance(df, pd.DataFrame)):
            raise(TypeError,'df must be a DataFrame not a ' + str(type(df)))

        self.__df = df
        self.__maximum_cardinality_for_one_hot_encode = 10
    
        # The converted dataframe
        self.__converted = None
        # Mapping from destination column to source column
        self.__colmap = None 
        # Feature or features the column corresponds to in the source column
        self.__featuremap = None
    
    @property
    def maximum_cardinality_for_one_hot_encode(self):
        return self.__maximum_cardinality_for_one_hot_encode
    
    @maximum_cardinality_for_one_hot_encode.setter
    def maximum_cardinality_for_one_hot_encode(self, newval):
        self.__maximum_cardinality_for_one_hot_encode = newval
        
    
    # xgboost needs the entire frame to be numeric.
    # So we'll convert strings to either one-hot or (if the cardinality is excessive for 1-hot) into a numerical label per unique string.
    def ConvertForXGBoost(self):
    
        self.__converted = pd.DataFrame()
        self.__colmap = dict()   
        self.__featuremap = dict()
    
        #print('Converting...')
        for col in self.__df.columns:
            #print('Column : ' + col)
            #print('Type: ' + str(type(self.__df[col][0])))            
            cardinality = len(self.__df[col].unique())
            #print('Cardinality: ' + str(cardinality))
            
            # If string, one-hot (if not too many unique values) or feature encode.
            if isinstance(self.__df[col][0], str):
                    
                    # Feature
                    label_encoder = LabelEncoder()
                    feature = label_encoder.fit_transform(self.__df[col])
                                    
                    if cardinality > self.__maximum_cardinality_for_one_hot_encode:
                        self.__converted[col] = feature
                        self.__colmap[col] = col
                        self.__featuremap[col] = dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))

                    else:
                        # One-hot
                        feature = feature.reshape(self.__df.shape[0], 1)
                        onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
                        feature = onehot_encoder.fit_transform(feature)
                        for f in range(feature.shape[1]):
                            self.__converted[col + '_' + str(f)] = feature[:,f]
                            self.__colmap[col + '_' + str(f)] = col
                            self.__featuremap[col + '_' + str(f)] = label_encoder.classes_[f]

            else:
                # Already numeric, copy directly.
                self.__converted[col] = self.__df[col]
                self.__colmap[col] = col
                self.__featuremap[col] = None
            
        return self.__converted, self.__colmap, self.__featuremap
        