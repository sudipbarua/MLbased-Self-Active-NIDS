from data_preparation import *
from unsupervised_labeling import UnsupervisedLabeling
import pandas as pd
import numpy as np
import yaml
import random
from sklearn import metrics
import os
from datetime import datetime
import errno
import pickle
from CNN import CNN
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from methods import Methods

def main():
    print("\n Running script...\n")
    # Read data
    ctu13 = pd.read_csv('./complete_all_scenarios_NoTransfer_withHeader_withLabel-26-1-2021-bak.csv').sample(n=300)
    ctu13 = DataPrep().label_converter(ctu13)
    ctu13 = DataPrep().dataset_cleaning(ctu13)
    ctu13 = ctu13.drop(['sTtl'], axis=1)

    val_batch = ctu13.tail(n=int(len(ctu13) * 0.3))
    # ctu13 = ctu13.drop(index=val_batch.index)
    X = ctu13.drop(['Label'], axis=1)
    Y = ctu13["Label"]
    x_test = val_batch.drop(['Label'], axis=1)
    y_test = val_batch["Label"]


    # with open("param_distribution.yaml", "r") as stream:
    #     try:
    #         parsed_yaml_file = yaml.safe_load(stream)
    #     except yaml.YAMLError as exc:
    #         print(exc)
    # param_clf = parsed_yaml_file['HDBSCAN']
    # param_dicts = []
    # model_score = []
    #
    # for i in range(100):
    #     temp_dict = {}
    #     for key in param_clf:
    #         rand_val = random.sample(param_clf[key], 1)  # returns list
    #         temp_dict[key] = rand_val[0]
    #     param_dicts.append(temp_dict)
    # for idx, params in enumerate(param_dicts):
    #     print(params)
    #     ul = UnsupervisedLabeling(model_name='HDBSCAN')
    #     ctu13_nl = ul.get_new_labels(df_scaled=ctu13,param=params)
    #     f1 = metrics.f1_score(ctu13_nl['Label'], ctu13_nl['new_label'])
    #     print(f1)
    #     model_score.append([params, f1, ul.n_query])
    #
    # headers = ['Model_setting', 'Score', "queries"]
    # model_acc_df = pd.DataFrame(np.asarray(model_score), columns=headers)
    # Methods().gen_file(model_acc_df, "model_score.csv", path="./results")
    # dir_path = Methods().gen_folder("./results", f"AL_Seq{datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')}")
    # ul = UnsupervisedLabeling(model_name='HDBSCAN', tune_search=2)
    # ctu13_nl = ul.get_new_labels(df_scaled=ctu13)
    # with open(f'{dir_path}/DBSCAN.pkl', 'wb') as f:
    #     pickle.dump(ul.model, f)
    #
    # # and later you can load it
    # with open(f'{dir_path}/DBSCAN.pkl', 'rb') as f:
    #     model2 = pickle.load(f)
    #
    path = "./results/AL_Seq_2022-02-06_15-13-12/Iteration_train_1/AL_EnS_avg/Retrain_1"
    model_cnn = CNN.model_cnn()
    model = model_cnn.load_cnn_model(path, "/model_cnn_retrain-1_fold-train_1")
    model_cnn.model.summary()
    print("end")


if __name__ == '__main__':
    main()