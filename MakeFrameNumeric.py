import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


class MakeFrameNumeric:

    def __init__(self, df):
         
        if( not isinstance(df, pd.DataFrame)):
            raise(TypeError,'df must be a DataFrame not a ' + str(type(df)))

        self.__df = df
        self.__maximum_cardinality_for_one_hot_encode = 3
    
    
    
    @property
    def maximum_cardinality_for_one_hot_encode(self):
        return self.__maximum_cardinality_for_one_hot_encode
    
    @maximum_cardinality_for_one_hot_encode.setter
    def maximum_cardinality_for_one_hot_encode(self, newval):
        self.__maximum_cardinality_for_one_hot_encode = newval
        
    
    # xgboost needs the entire frame to be numeric.
    # So we'll convert strings to either one-hot or (if the cardinality is excessive for 1-hot) into a numerical label per unique string.
    def ConvertForXGBoost(self):
    
        converted = pd.DataFrame()
    
        print('Converting...')
        for col in self.__df.columns:
            print('Column : ' + col)
            print('Type: ' + str(type(self.__df[col][0])))
            
            cardinality = len(self.__df[col].unique())
            print('Cardinality: ' + str(cardinality))
            
            # If string, one-hot (if not too many unique values) or feature encode.
            if isinstance(self.__df[col][0], str):
                    
                    # Feature
                    label_encoder = LabelEncoder()
                    feature = label_encoder.fit_transform(self.__df[col])
                    if cardinality > self.__maximum_cardinality_for_one_hot_encode:
                        converted[col] = feature
                    else:
                        # One-hot
                        feature = feature.reshape(self.__df.shape[0], 1)
                        onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
                        feature = onehot_encoder.fit_transform(feature)
                        print(feature)
                        for f in range(feature.shape[1]):
                            converted[col + '_' + str(f)] = feature[:,f]
            else:
                # Already numeric, copy directly.
                converted[col] = self.__df[col]
            
        return converted