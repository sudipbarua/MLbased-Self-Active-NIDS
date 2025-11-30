"""
This class contains some necessary method used by all other modules
"""
import errno
import os
import seaborn as sns
import matplotlib as plt
import yaml
from datetime import datetime
from sklearn import metrics


class Methods:
    def __init__(self):
        pass

    def gen_file(self, df, filename, path, dated_sub_dir=True):
        """
        df: dataframe to be saved
        filename: file's name in string format (.csv)
        path: path name (as string) in which the folder will be created and the file be saved
        """
        if dated_sub_dir is True:
            file_dir = os.path.join(path, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
            try:
                os.makedirs(file_dir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise  # This was not a "directory exist" error..
            file_path = os.path.join(file_dir, filename)
        else:
            file_path = os.path.join(path, filename)
        df.to_csv(file_path)
        return file_path

    def gen_folder(self, path, name):
        dir_path = os.path.join(path, name)
        try:
            os.makedirs(dir_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise  # This was not a "directory exist" error..

        return dir_path

    def get_eval(self, x, y, model, train_set, feature_set='All'):
        """
        generates the evaluation scores (Precision, Recall, F1 score)
        Returns scores in a list
        Include feature set if feature section is performed.
        Default 'All' (when all features are considered)
        """
        f1 = metrics.f1_score(x, y)
        p = metrics.precision_score(x, y)
        r = metrics.recall_score(x, y)
        print(f"\nclassification report for {model} {train_set}:"
              f"\n Recall: {r}"
              f"\n Precision: {p}"
              f"\n F1 score: {f1}")
        ev = [train_set, feature_set, model, p, r, f1]
        return ev

    def parse_config(self, file_name):
        # load the configuration file
        with open(file_name, "r") as stream:
            try:
                param_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        return param_dict

    def is_unique(self, df):
        arr = df.to_numpy()
        return (arr[0] == arr).all()

    def find_majority_elem(self, k, th):
        myMap = {}
        maj_elem_cnt = ['', 0]  # (occurring element, occurrences)
        # th = 0.50  # Set a threshold for majority selection (90%)
        for n in k:
            if n in myMap:
                myMap[n] += 1
            else:
                myMap[n] = 1
            # Keep track of maximum on the go
            if myMap[n] > maj_elem_cnt[1]: maj_elem_cnt = [n, myMap[n]]
        if maj_elem_cnt[1]/len(k.index) < th:
            maj_elem = None
        else:
            maj_elem = maj_elem_cnt[0]
        return maj_elem

