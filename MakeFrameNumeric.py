import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


class FrameTools:

    def __init__(self, df):
    
        self.df = df
        self.maximum_cardinality_for_one_hot_encode = 3
       
    def listcols(self):
        print(self.df.columns)


    # xgboost needs the entire array to be numeric.
    # So we'll convert strings to either one-hot or (if the cardinality is excessive for 1-hot) into a numerical label per unique string.
    def convert_for_xgboost(self):
    
        converted = pd.DataFrame()
    
        print('Converting...')
        for col in self.df.columns:
            print('Column : ' + col)
            print('Type: ' + str(type(self.df[col][0])))
            print('Length: ' + str(len(self.df[col].unique())))
            
            # If string, one-hot (if not too many unique values) or feature encode.
            if isinstance(self.df[col][0], str):
                    
                    # Feature
                    label_encoder = LabelEncoder()
                    feature = label_encoder.fit_transform(self.df[col])
                    if len(self.df[col].unique()) > self.maximum_cardinality_for_one_hot_encode:
                        converted[col] = feature
                    else:
                        # One-hot
                        feature = feature.reshape(self.df.shape[0], 1)
                        onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
                        feature = onehot_encoder.fit_transform(feature)
                        print(feature)
                        for f in range(feature.shape[1]):
                            converted[col + '_' + str(f)] = feature[:,f]
            else:
                # Already numeric, copy directly.
                converted[col] = self.df[col]
            
        print(converted)