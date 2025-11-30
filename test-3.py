from unsupervised_labeling import UnsupervisedLabeling
from data_preparation import DataPrep
import pandas as pd
from sklearn import metrics
from methods import Methods
import numpy as np


result = []
for i in range(1,6):
    ctu13 = pd.read_csv('complete_all_scenarios_NoTransfer_withHeader_withLabel-26-1-2021-bak.csv').sample(n=100)
    ctu13 = DataPrep().label_converter(ctu13)
    ctu13 = DataPrep().dataset_cleaning(ctu13)

    clusterer = UnsupervisedLabeling(model_name='HDBSCAN')
    ctu13_nl = clusterer.get_new_labels(ctu13)
    pr = metrics.precision_score(ctu13_nl['Label'], ctu13_nl['new_label'])
    recall = metrics.recall_score(ctu13_nl['Label'], ctu13_nl['new_label'])
    f1 = metrics.f1_score(ctu13_nl['Label'], ctu13_nl['new_label'])
    n_query = clusterer.n_query
    n_cl = clusterer.n_clusters
    r = [i, pr, recall, f1, n_query]
    result.append(r)

    df_path = Methods().gen_file(ctu13_nl, f"df_nl_{i}.csv", "./results/hdbscan", False)

headers = ['Iter', 'Precision', 'recall', 'F1', 'n_query']
results = pd.DataFrame(np.asarray(result), columns=headers)
result_path = Methods().gen_file(results, "results_hdbscan.csv", './results/hdbscan', False)