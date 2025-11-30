import os
from datetime import datetime

import multiprocessing

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from unsupervised_labeling import *
from data_preparation import *
from data_splitting import *
from active_learner import *
from CNN import CNN
from methods import Methods
import model_fitter

def main():
    print("\n Running script...\n")
    # generate main folder
    dir_path = Methods().gen_folder("./results", f"AL_Seq{datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')}")
    # generate report.csv file
    report_path = Methods().gen_file(df=pd.DataFrame(list()),
                                     filename=f"report{datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')}.csv",
                                     path=dir_path, dated_sub_dir=False)
    # Read data
    ctu13 = pd.read_csv('./complete_all_scenarios_NoTransfer_withHeader_withLabel-26-1-2021-bak.csv').sample(n=12000)
    ctu13 = ctu13.drop(['sTtl'], axis=1)
    # Convert the ground truth Labels to numeric values
    ctu13_1 = DataPrep().label_converter(ctu13)
    ds_1 = DataSplit(n_chunks=3,
                     n_chunk_train=2,
                     n_chunk_test=1)
    train_df, test_df = ds_1.dataset_spit(ctu13_1, report_path)

    batch_sz = 3000
    init_batch_sz = 6000
    result_df = pd.DataFrame()

    # Iterate over different datasets
    for tr, ts in zip(train_df, test_df):
        itr_path = Methods().gen_folder(dir_path, f"Iteration_{tr}")
        print(f"------------------------------Begin Iteration: {tr}-------------------------------", file=open(report_path, 'a'))
        print(f"*************************************Dataset**************************************", file=open(report_path, 'a'))
        # Scale dataset
        ctu13_train = DataPrep().dataset_cleaning(train_df[tr])
        train_file_path = Methods().gen_file(ctu13_train, f"{tr}.csv", itr_path, False)  # save training set
        ctu13_test = DataPrep().dataset_cleaning(test_df[ts])
        test_file_path = Methods().gen_file(ctu13_test, f"{ts}.csv", itr_path, False)  # save test set
        print(f"Train Dataset Dimension: {train_df[tr].shape}", file=open(report_path, 'a'))
        print(f"Test Dataset Dimension: {test_df[ts].shape}", file=open(report_path, 'a'))
        print(f"Initial training batch size: {init_batch_sz}", file=open(report_path, 'a'))
        print(f"Normal training batch size: {batch_sz}", file=open(report_path, 'a'))
        # al_model_rf = model_fitter(batch_sz, init_batch_sz, True, ctu13_train, ctu13_test, "AL_RF_US", tr, itr_path, report_path)
        al_model_ens_avg = model_fitter.model_fitter(batch_sz, init_batch_sz, True, ctu13_train, ctu13_test, "AL_EnS_avg", tr, itr_path, report_path)
        # al_model_knn = model_fitter(batch_sz, init_batch_sz, True, ctu13_train, ctu13_test, "AL_KNN_US", tr, itr_path, report_path)
        dal_model_us = model_fitter.model_fitter(batch_sz, init_batch_sz, True, ctu13_train, ctu13_test, "DAL_US", tr, itr_path, report_path)
        # al_model_svmC2H = model_fitter(batch_sz, init_batch_sz, True, ctu13_train, ctu13_test, "AL_SVM_C2H", tr, itr_path, report_path)
        # dal_model_tob = model_fitter(batch_sz, init_batch_sz, True, ctu13_train, ctu13_test, "DAL_TOB", tr, itr_path, report_path)
        # pl_model_rf = model_fitter(batch_sz, init_batch_sz, False, ctu13_train, ctu13_test, 'PL_RF', tr, itr_path, report_path)
        # pl_model_knn = model_fitter(batch_sz, init_batch_sz, False, ctu13_train, ctu13_test, 'PL_KNN', tr, itr_path, report_path)
        # dpl_rs_model = model_fitter(batch_sz, init_batch_sz, False, ctu13_train, ctu13_test, "DPL_RS", tr, itr_path, report_path)
        # dpl_tob_model = model_fitter(batch_sz, init_batch_sz, False, ctu13_train, ctu13_test, "DPL_TOB", tr, itr_path, report_path)


        result_df = result_df.append([al_model_ens_avg.results_df, dal_model_us.results_df], ignore_index=True)


        Methods().gen_file(result_df, "results.csv", path=itr_path, dated_sub_dir=False)
        print(f"-------------------------End Iteration: {tr}-----------------------------", file=open(report_path, 'a'))

    model_fitter.plot_all(dir_path, result_df)



if __name__ == '__main__':
    main()