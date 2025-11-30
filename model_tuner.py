import random
from sklearn import metrics
import pandas as pd
import numpy as np
import hdbscan
import itertools

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from methods import Methods


class ModelTuner:
    def __init__(self, model_name, n_search=20, grid_search=False, score_dir=None):
        self.param_dicts = []  # List of the randomly selected parameter sets
        self.best_params = {}  # best parameter set
        self.model_name = model_name
        self.n_search = n_search
        self.grid_search = grid_search
        self.score_dir = score_dir

    def get_random_param_set(self):
        for i in range(self.n_search):
            temp_dict = {}
            for key in self.param_dist:
                rand_val = random.sample(self.param_dist[key], 1)  # returns list
                temp_dict[key] = rand_val[0]
            self.param_dicts.append(temp_dict)

    def get_grid_param_set(self):
        tmp_lst = []
        for key, lst in self.param_dist.items():
            tmp_lst.append(lst)
        combo_lst = list(itertools.product(*tmp_lst))
        for item in combo_lst:
            temp_dict = {}
            for j, key in zip(item, self.param_dist):
                temp_dict[key] = j
            self.param_dicts.append(temp_dict)

    def fit_model(self):
        if self.model_name == "HDBSCAN":
            self.model.fit(self.val_x)
            self.score = self.model.relative_validity_
            labels = np.resize(self.model.labels_, (len(self.model.labels_), 1))
            self.n_clusters = np.amax(labels) + 1
        else:
            self.model.fit(self.val_x, self.val_y)
            y_pred = self.model.predict(self.x_test)
            self.score = metrics.f1_score(self.y_test, y_pred)

    def random_search(self, param_dist, val_x, val_y=None, x_test=None, y_test=None, save_score=False):
        self.val_x = val_x
        self.val_y = val_y
        self.x_test = x_test
        self.y_test = y_test
        self.param_dist = param_dist
        score_lst = []
        self.model_score = []
        if self.grid_search == True:
            self.get_grid_param_set()
        else:
            self.get_random_param_set()
        for idx, params in enumerate(self.param_dicts):
            self.build_model(params)
            try:
                self.fit_model()
            except ValueError as err:
                print(err, "\nPlease provide valid input in params, val_y, x_test, y_test in ClassifierTuning.random_search()")
                break
            score_lst.append(self.score)
            if self.model_name == "HDBSCAN":
                self.model_score.append([params, self.score, self.n_clusters])
            else:
                self.model_score.append([params, self.score])
        # Get the best parameter dictionary
        self.best_params = self.param_dicts[score_lst.index(max(score_lst))]
        self.build_model(self.best_params)
        if save_score:
            self.save_model_score()
        
    def build_model(self, params):
        if self.model_name == "SVC":
            self.model = SVC(**params)
        elif self.model_name == "KNeighborsClassifier":
            self.model = KNeighborsClassifier(**params)
        elif self.model_name == "RandomForestClassifier":
            self.model = RandomForestClassifier(**params)
        elif self.model_name == "HDBSCAN":
            self.model = hdbscan.HDBSCAN(**params)

    def save_model_score(self):
        if self.model_name == "HDBSCAN":
            headers = ['Model_setting', 'Score', 'n_clusters']
        else:
            headers = ['Model_setting', 'Score']
        model_acc_df = pd.DataFrame(np.asarray(self.model_score), columns=headers)
        Methods().gen_file(model_acc_df, "model_score.csv", path=self.score_dir, dated_sub_dir=False)
        print("end")
