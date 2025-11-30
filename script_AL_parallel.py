import os
from datetime import datetime

import multiprocessing

import seaborn as sns
from pandas.errors import DtypeWarning

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
warnings.simplefilter(action="ignore", category=DtypeWarning)


def model_fitter(batch_sz, init_batch_sz, learner_typ, x, y, ml_method, tr_set, itr_path, report_path):
    """
    Helper function for multiprocessing
    :param batch_sz: Active learner batch size
    :param init_batch_sz: Initial training batch size
    :param learner_typ: True if active learner, False if passive
    :param x: Train dataset
    :param y: Test dataset
    :param ml_method: Method to used for corresponding active learning scenario
    :param tr_set: Train dataset marker
    :return: Generated result of each AL Scenario
    """
    al_method = None
    cnn_model = None
    clf_model = None
    if learner_typ is False:
        if ml_method == 'PL_RF':
            clf_model = RandomForestClassifier(n_estimators=1000)
        elif ml_method == 'PL_KNN':
            clf_model = KNeighborsClassifier()
        elif ml_method == 'DPL_RS':
            cnn_model = CNN.model_cnn()
        elif ml_method == 'DPL_TOB':
            al_method = "CNN_TOB"
            cnn_model = CNN.model_cnn()
    elif learner_typ is True:
        if ml_method == 'AL_SVM_US':
            clf_model = SVC(probability=True)
        if ml_method == 'AL_LR_US':
            clf_model = LogisticRegression(C=10, penalty='l2', solver='newton-cg')
        elif ml_method == "AL_RF_US":
            clf_model = RandomForestClassifier(n_estimators=1000)
        elif ml_method == "AL_KNN_US":
            clf_model = KNeighborsClassifier()
        elif ml_method == "AL_EnS_Maj":
            lr_EnS_maj = LogisticRegression(C=10, penalty='l2', solver='newton-cg')
            rf_EnS_maj = RandomForestClassifier(n_estimators=1000)
            knn_EnS_maj = KNeighborsClassifier()
            clf_model = [("LR_EnS_maj", lr_EnS_maj), ("RF_EnS_maj", rf_EnS_maj), ("KNN_EnS_maj", knn_EnS_maj)]
            al_method = "EnS_majority"
        elif ml_method == 'AL_EnS_avg':
            cnn_EnS_avg = CNN.model_cnn()
            rf_EnS_avg = RandomForestClassifier(n_estimators=1000)
            knn_EnS_avg = KNeighborsClassifier()
            clf_model = {"CNN": cnn_EnS_avg, "RF_EnS_avg": rf_EnS_avg, "KNN_EnS_avg": knn_EnS_avg}  # insert "CNN" as key if classifier is a CNN model
            al_method = "EnS_avg"
        elif ml_method == 'AL_SVM_C2H':
            clf_model = SVC()
            al_method = "SVM_C2H"
        elif ml_method == 'DAL_US':
            cnn_model = CNN.model_cnn()
        elif ml_method == 'DAL_TOB':
            al_method = "CNN_TOB"
            cnn_model = CNN.model_cnn()
        elif ml_method == 'DAL_CL_US':
            al_method = "CNN_US_CL"
            cnn_model = CNN.model_cnn()
    model_dir = Methods().gen_folder(itr_path, ml_method)  # generate model directory
    al_model = ActiveLearner(al_method = al_method,
                             batch_size=batch_sz,
                             init_batch_sz=init_batch_sz,
                             active_learner=learner_typ,
                             model_dir=model_dir, report_path=report_path)
    al_model.fit(train_df=x,
                 test_df=y,
                 clf_models=clf_model,
                 cnn_model=cnn_model,
                 ml_method=ml_method, tr_set=tr_set)
    return al_model.results_df, al_model.train_time_tot, al_model.test_time_tot, al_model.tune_time_tot

def main():
    print("\n Running script...\n")
    # generate main folder
    dir_path = Methods().gen_folder("./results", f"AL_Seq{datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')}")
    # generate report.csv file
    report_path = Methods().gen_file(df=pd.DataFrame(list()),
                                     filename=f"report{datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')}.csv",
                                     path=dir_path, dated_sub_dir=False)
    # Read data
    ctu13 = pd.read_csv('./complete_all_scenarios_NoTransfer_withHeader_withLabel-26-1-2021-bak.csv').sample(n=24000)
    ctu13 = ctu13.drop(['sTtl'], axis=1)
    # Convert the ground truth Labels to numeric values
    ctu13_1 = DataPrep().label_converter(ctu13)
    ds_1 = DataSplit(n_chunks=4,
                     n_chunk_train=3,
                     n_chunk_test=1)
    train_df, test_df = ds_1.dataset_spit(ctu13_1, report_path)

    batch_sz = 3000
    init_batch_sz = 6000
    result_df = pd.DataFrame()

    # Iterate over different datasets
    for tr, ts in zip(train_df, test_df):
        itr_path = Methods().gen_folder(dir_path, f"Iteration_{tr}")
        print(f"------------------------------Begin Iteration: {tr}-------------------------------",
              file=open(report_path, 'a'))
        print(f"*************************************Dataset**************************************",
              file=open(report_path, 'a'))
        # Scale dataset
        ctu13_train = DataPrep().dataset_cleaning(train_df[tr])
        train_file_path = Methods().gen_file(ctu13_train, f"{tr}.csv", itr_path, False)  # save training set
        ctu13_test = DataPrep().dataset_cleaning(test_df[ts])
        test_file_path = Methods().gen_file(ctu13_test, f"{ts}.csv", itr_path, False)  # save test set

        param_pool = [
            # (batch_sz, init_batch_sz, False, ctu13_train, ctu13_test, 'PL_RF', tr),
            # (batch_sz, init_batch_sz, False, ctu13_train, ctu13_test, 'PL_KNN', tr),
            # (batch_sz, init_batch_sz, True, ctu13_train, ctu13_test, 'AL_LR_US', tr),
            # (batch_sz, init_batch_sz, True, ctu13_train, ctu13_test, "AL_RF_US", tr),
            # (batch_sz, init_batch_sz, True, ctu13_train, ctu13_test, "AL_KNN_US", tr),
            # (batch_sz, init_batch_sz, True, ctu13_train, ctu13_test, "AL_EnS_Maj", tr),
            (batch_sz, init_batch_sz, True, ctu13_train, ctu13_test, "AL_EnS_avg", tr, itr_path, report_path),
            # (batch_sz, init_batch_sz, True, ctu13_train, ctu13_test, "AL_SVM_C2H", tr),
            # (batch_sz, init_batch_sz, False, ctu13_train, ctu13_test, "DPL_RS", tr),
            # (batch_sz, init_batch_sz, False, ctu13_train, ctu13_test, "DPL_TOB", tr),
            # (batch_sz, init_batch_sz, True, ctu13_train, ctu13_test, "DAL_US", tr),
            # (batch_sz, init_batch_sz, True, ctu13_train, ctu13_test, "DAL_TOB", tr),
            # (batch_sz, init_batch_sz, True, ctu13_train, ctu13_test, "DAL_CL_US", tr)
        ]
        process_pool = multiprocessing.Pool()
        [(result_df, train_time_tot, test_time_tot, tune_time_tot)] = process_pool.starmap(model_fitter, param_pool)
        result_df = result_df.append(result_df, ignore_index=True)
        result_df = result_df.astype({
            'Number_of_Samples_trained': 'float64',
            'Normal_traffic_label_err_rate': 'float64',
            'Botnet_label_err_rate': 'float64',
            'F1': 'float64',
            'N_Retrains': 'float64',
            'Expert_time': 'float64',
            'Queries_to_Oracle': 'float64',
            'Precision': 'float64', 'Recall': 'float64',
            'Training_Time': 'float64', 'Testing_Time': 'float64', 'Tuning_Time': 'float64'
        })
        Methods().gen_file(result_df,"active_learner_results.csv", path="./results")
        print(f"-----------------------End Iteration: {tr}-----------------------------")
    print("\nSaving plots...")
    result_df1 = result_df.copy()
    result_df1.drop(result_df1[result_df1.N_Retrains < 2].index, inplace=True)
    plot_save_results(result_df1, x=result_df1['Number_of_Samples_trained'],
                      y=result_df1['Botnet_label_err_rate'],
                      hue=result_df1['ML_method'], plot_name="Botnet_Labeling_Error_vs_nSamples")

    plot_save_results(result_df1, x=result_df1['Number_of_Samples_trained'],
                      y=result_df1['Normal_traffic_label_err_rate'],
                      hue=result_df1['ML_method'], plot_name="Normal_Traffic_Error_vs_nSamples")

    plot_save_results(result_df, x=result_df['N_Retrains'],
                      y=result_df['F1'],
                      hue=result_df['ML_method'], plot_name="F1_vs_nRetrains")

    plot_save_results(result_df, x=result_df['Expert_time'],
                      y=result_df['F1'],
                      hue=result_df['ML_method'], plot_name="F1_vs_ExpertTime")

    plot_save_results(result_df, x=result_df['Queries_to_Oracle'],
                      y=result_df['F1'],
                      hue=result_df['ML_method'], plot_name="F1_vs_nQueries")
    # Prepare new df containing rows with max values of queries
    result_df2 = result_df[result_df.N_Retrains == result_df.N_Retrains.max()]
    bar_plot_save(result_df2, x=result_df2['ML_method'], y=result_df2['Queries_to_Oracle'], hue=None,
                  plot_name="N_queries_vs_learning algorithm")

    df4 = pd.DataFrame()
    df4['ML_method'] = result_df2['ML_method']
    df4['n_Norm_Traffic_label_err'] = result_df2['n_Norm_Traffic_label_err']
    df4['n_Bot_label_err'] = result_df2['n_Bot_label_err']
    # combine 'n_Norm_Traffic_label_err' and 'n_Bot_label_err' to single 'cols' and 'vals' columns
    dfmelt = df4.melt('ML_method', var_name='cols', value_name='vals')
    # Exclude the passive learning results
    dfmelt.drop(dfmelt[dfmelt.ML_method.str.contains('PL_')].index, inplace=True)
    bar_plot_save(x=dfmelt['ML_method'], y=dfmelt['vals'], hue='cols', df=dfmelt, plot_name='Error_vs_learning algorithm')

    print("\nDone...")


def plot_save_results(df, x, y, hue, plot_name):
    sns.lineplot(x=x, y=y, hue=hue, data=df, style=hue, markers=True, dashes=True, err_style="bars", ci=5)
    path = os.path.join('./fig', '{}_{}.jpg'.format(plot_name, datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    fig = plt.gcf()
    fig.set_size_inches(15, 5)
    fig.savefig(path, dpi=400)
    plt.show()

def bar_plot_save(df, x, y, hue, plot_name):
    sns.barplot(x=x, y=y, hue=hue, data=df)
    path = os.path.join('./fig', '{}_{}.jpg'.format(plot_name, datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    plt.xticks(rotation=45, horizontalalignment='right', fontweight='light', fontsize='x-large')
    fig = plt.gcf()
    fig.set_size_inches(10, 15)
    fig.savefig(path, dpi=400)
    plt.show()


if __name__ == '__main__':
    main()