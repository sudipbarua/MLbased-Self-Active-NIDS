from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
import csv
from datetime import datetime
from scipy.stats import loguniform
from skopt import BayesSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from data_preparation import *


def main():
    print("Loading data....")
    # load dataset
    ctu13 = pd.read_csv('./complete_all_scenarios_NoTransfer_withHeader_withLabel-26-1-2021-bak.csv').sample(n=5000)
    ctu13 = DataPrep().label_converter(ctu13.drop('sTtl', axis=1))
    ctu13_scaled = DataPrep().dataset_cleaning(ctu13)
    # split into input and output elements
    X, y = ctu13_scaled.drop('Label', axis=1), ctu13_scaled['Label']
    # define search space
    params = {'C': (1e-2, 10.0, 'log-uniform'),
              'gamma': (1e-3, 10.0, 'log-uniform'),
              'degree': (1, 5),
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
    # Parameter space for grid search
    param_grid = {'C': [1e-2, 1e-1, 1, 10],
                  'gamma': [1e-3, 1e-2, 1e-1, 1, 10],
                  'degree': [1,2,3,4,5],
                  'kernel': ['linear', 'poly','rbf']}
    # Perform Bayes Search
    bayes_search(params, X, y)
    # perform grid search
    grid_search(param_grid, X, y)
    # Perform Randomized search
    random_search(param_grid, X, y)


def bayes_search(param_space, X, y):
    print("Performing Bayes search...")
    # define evaluation split
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    # define the search
    search = BayesSearchCV(estimator=SVC(), search_spaces=param_space, n_jobs=-1, cv=cv)
    # perform the search
    search.fit(X, y)
    # report the best result
    print(search.best_score_)
    best_params = dict(search.best_params_)
    save_best_params(best_params, "Bayes Search", list(param_space.keys()))


def grid_search(param_space, X, y):
    print("Performing Grid search...")
    # define evaluation split
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    # define the search
    search = GridSearchCV(estimator=SVC(), param_grid=param_space, n_jobs=-1, cv=cv)
    # perform the search
    search.fit(X, y)
    # report the best result
    print(search.best_score_)
    save_best_params(search.best_params_, "Grid Search", list(param_space.keys()))


def random_search(param_space, X, y):
    print("Performing Random search...")
    # define evaluation split
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    # define the search
    search = RandomizedSearchCV(estimator=SVC(), param_distributions=param_space, n_jobs=-1, cv=cv)
    # perform the search
    search.fit(X, y)
    # report the best result
    print(search.best_score_)
    save_best_params(search.best_params_, "Randomized Search", list(param_space.keys()))


def save_best_params(params, search_typ, keys):
    print("Saving parameter values...")
    val_dict = {k: params[k] for k in keys}
    print("Best parameter values with {}: \n{}".format(search_typ, val_dict))
    with open('best_parameter_values.csv', 'a+', newline='') as file:
        # creating the csv writer
        file_write = csv.DictWriter(file, fieldnames=val_dict.keys())
        # storing current date and time
        current_date_time = datetime.now()
        val_dict.update(time=current_date_time, search_type=search_typ)
        file_write.writerow(val_dict)
        file.close()
    print("Done.")


if __name__ == '__main__':
    main()
