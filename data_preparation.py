"""
This module prepares the dataframe
1. label_converter - converts the string labels to binary labels
2. dataset_cleaning - converts all values that are not int/float to be filled with nan replace with median values
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class DataPrep:

    def __init__(self):
        pass

    def label_converter(self, df):
        mask_0 = df['Label'].str.contains('Normal')
        mask_1 = df['Label'].str.contains('Botnet')

        # Replace the label of normal traffic with 0
        df.loc[mask_0, 'Label'] = 0

        # Replace the label of abnormal/botnet traffic with 1
        df.loc[mask_1, 'Label'] = 1

        return df
    # End of label_converter function

    def dataset_cleaning(self, df):
        cols = df.columns
        for col in cols:
            # convert all values to number.(in the case any string is included)
            # the error value force to put Nan if there is any error to convert string to value
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.replace([np.inf, -np.inf], np.nan, inplace=True)  # replace infinity values with NaNs
        # Fill nans with median values
        df.fillna(df.median(), inplace=True)
        scaler = MinMaxScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=cols)
        return df_scaled
    # End of method dataset_cleaning
