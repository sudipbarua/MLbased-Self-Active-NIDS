"""
Shuffles the dataset.
Splits the dataset into equal chunks according to the given n_chunks.
Then combines the chunks according to n_chunk_train, n_chunk_test to create train and test datasets.
Saves the train or test set in "./data_splitted" dir if "save_file=True" (Default = False)
"""

import numpy as np
import pandas as pd
from itertools import combinations
from sklearn import utils


class DataSplit:

    def __init__(self,
                 n_chunks=5,
                 n_chunk_train=3,
                 n_chunk_test=2):
        self.n_chunks = n_chunks
        self.n_chunk_train = n_chunk_train
        self.n_chunk_test = n_chunk_test

    def train_test_dict(self, n_chunks, key):
        """
        args : splitted dataset, number of chunks for either training or testing dataset and key = 'train' or 'test'
        returns a dictionary of train or test dataset
        """
        # itertool combination does not return a list, rather tuples. So we convert it to list
        comb = list(combinations(self.df_splitted, n_chunks))

        train_test_dict_x = {}

        if key == 'train':
            for i in range(0, len(comb)):
                z = pd.DataFrame()  # create a new dataframe
                y = comb[i]
                for item in y:
                    z = pd.concat([z, item])  # concat items
                # train data
                train_test_dict_x["{}_{}".format(key, i + 1)] = z

        elif key == 'test':
            for j in range(len(comb)-1, -1, -1):
                z = pd.DataFrame()  # create a new dataframe
                y = comb[j]
                for item in y:
                    z = pd.concat([z, item])  # concat items
                # test data
                train_test_dict_x["{}_{}".format(key, len(comb)-j)] = z
        return train_test_dict_x

    def dataset_spit(self, df_labeled, report_path):
        print("-----------------Splitting Dataset--------------------", file=open(report_path, 'a'))
        print(f"Number of Chunks: {self.n_chunks}", file=open(report_path, 'a'))
        print(f"Number of Chunks for train: {self.n_chunk_train}", file=open(report_path, 'a'))
        print(f"Number of Chunks for test: {self.n_chunk_test}", file=open(report_path, 'a'))
        df_shuffled = utils.shuffle(df_labeled)
        # split data
        self.df_splitted = np.array_split(df_shuffled, self.n_chunks)

        # create train dataset dictionary
        train_x_dict = self.train_test_dict(self.n_chunk_train, 'train')
        # create test dataset dictionary
        test_x_dict = self.train_test_dict(self.n_chunk_test, 'test')

        return train_x_dict, test_x_dict
    # End of method dataset_spit
