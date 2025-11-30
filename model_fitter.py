from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from CNN import CNN
from active_self_learner import ActiveSelfLearner
from active_learner_sample_initialize import ActiveLearnerSampleInit
from active_learner import ActiveLearner
from methods import Methods
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import pandas as pd

def main():
    config_dict = Methods().parse_config("config_directories.yaml")  # parse the config and dir file
    resul_df = pd.DataFrame()
    for k, path in config_dict.items():
        # filter out the paths of "result" dfs
        if k.startswith('results'):
            temp_df = pd.read_csv(path, index_col=0)
            resul_df = resul_df.append(temp_df, ignore_index=True)
    result_path = Methods().gen_file(resul_df, "combined_results.csv",
                                     config_dict['main_directory'], dated_sub_dir=False)
    plot_all(config_dict['main_directory'], resul_df)

def model_fitter(batch_sz, init_batch_sz, learner_typ, x, y, ml_method, active_learner,
                 tr_set, itr_path, report_path, clusterer_name='HDBSCAN', load=False,
                 init_train_sample_sz=2000):
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
    cnn_param = {'input_shape': (28,1), 'learn_rate': 0.0001, 'activation': 'relu', 'layers': [128, 256]}
    if learner_typ is False:
        if ml_method == 'PL_RF':
            clf_model = RandomForestClassifier(n_estimators=600, n_jobs=-1)
        elif ml_method == 'PL_KNN':
            clf_model = KNeighborsClassifier(leaf_size=22, n_neighbors=10, p=1, weights='distance',
                                             algorithm='ball_tree', n_jobs=-1)
        elif ml_method == 'DPL_RS':
            cnn_model = CNN.model_cnn()
            cnn_model.model = cnn_model.build_cnn_model(**cnn_param)
        elif ml_method == 'DPL_TOB':
            al_method = "CNN_TOB"
            cnn_model = CNN.model_cnn()
            cnn_model.model = cnn_model.build_cnn_model(**cnn_param)
    elif learner_typ is True:
        if ml_method == 'AL_SVM_US':
            clf_model = SVC(C=250, gamma=80, probability=True)
        if ml_method == 'AL_LR_US':
            clf_model = LogisticRegression(C=10, penalty='l2', solver='newton-cg')
        elif ml_method == "AL_RF_US":
            clf_model = RandomForestClassifier(n_estimators=600, n_jobs=-1)
        elif ml_method == "AL_KNN_US":
            clf_model = KNeighborsClassifier(leaf_size=22, n_neighbors=10, p=1, weights='distance',
                                             algorithm='ball_tree', n_jobs=-1)
        elif ml_method == "AL_EnS_Maj":
            lr_EnS_maj = LogisticRegression(C=10, penalty='l2', solver='newton-cg')
            rf_EnS_maj = RandomForestClassifier(n_estimators=600, n_jobs=-1)
            knn_EnS_maj = KNeighborsClassifier(leaf_size=22, n_neighbors=10, p=1, weights='distance',
                                               algorithm='ball_tree', n_jobs=-1)
            clf_model = [("LR_EnS_maj", lr_EnS_maj), ("RF_EnS_maj", rf_EnS_maj), ("KNN_EnS_maj", knn_EnS_maj)]
            al_method = "EnS_majority"
        elif ml_method == 'AL_EnS_avg':
            cnn_EnS_avg = CNN.model_cnn()
            cnn_EnS_avg.model = cnn_EnS_avg.build_cnn_model(**cnn_param)
            rf_EnS_avg = RandomForestClassifier(n_estimators=600, n_jobs=-1)
            knn_EnS_avg = KNeighborsClassifier(leaf_size=22, n_neighbors=10, p=1, weights='distance',
                                               algorithm='ball_tree', n_jobs=-1)
            clf_model = {"CNN_EnS_avg": cnn_EnS_avg, "RF_EnS_avg": rf_EnS_avg,
                         "KNN_EnS_avg": knn_EnS_avg}  # insert "CNN" as key if classifier is a CNN model
            al_method = "EnS_avg"
        elif ml_method == 'AL_SVM_C2H':
            clf_model = SVC(C=250, gamma=80)
            al_method = "SVM_C2H"
        elif ml_method == 'DAL_US':
            cnn_model = CNN.model_cnn()
            cnn_model.model = cnn_model.build_cnn_model(**cnn_param)
        elif ml_method == 'DAL_TOB':
            al_method = "CNN_TOB"
            cnn_model = CNN.model_cnn()
            cnn_model.model = cnn_model.build_cnn_model(**cnn_param)
        elif ml_method == 'DAL_CL_US':
            al_method = "CNN_US_CL"
            cnn_model = CNN.model_cnn()
            cnn_model.model = cnn_model.build_cnn_model(**cnn_param)
    if load:
        model_dir = itr_path + f'/{ml_method}'
    else:
        model_dir = Methods().gen_folder(itr_path, ml_method)  # generate model directory
    if active_learner == "AL_2_stage":
        al_model = ActiveLearnerSampleInit(al_method=al_method,
                                           batch_size=batch_sz,
                                           init_batch_sz=init_batch_sz,
                                           active_learner=learner_typ, load=load,
                                           model_dir=model_dir, report_path=report_path,
                                           tune_search_count=20, clusterer_name=clusterer_name,
                                           init_train_sample_sz=init_train_sample_sz)
    elif active_learner == "AL_SL":
        al_model = ActiveSelfLearner(al_method=al_method,
                                 batch_size=batch_sz,
                                 init_batch_sz=init_batch_sz,
                                 active_learner=learner_typ, load=load,
                                 model_dir=model_dir, report_path=report_path,
                                 tune_search_count=20, clusterer_name=clusterer_name)
    else:
        al_model = ActiveLearner(al_method=al_method,
                                 batch_size=batch_sz,
                                 init_batch_sz=init_batch_sz,
                                 active_learner=learner_typ, load=load,
                                 model_dir=model_dir, report_path=report_path,
                                 tune_search_count=20, clusterer_name=clusterer_name)

    al_model.fit(train_df=x,
                 test_df=y,
                 clf_models=clf_model,
                 cnn_model=cnn_model,
                 ml_method=ml_method, tr_set=tr_set)
    print(f'Total test time: {al_model.test_time_tot}'
          f'\nTotal train time: {al_model.train_time_tot}'
          f'\nTotal tune time: {al_model.tune_time_tot}'
          f'\nNormal Traffic Label Error: {al_model.norm_label_err}'
          f'\nBotnet Label Error: {al_model.bot_label_err}'
          f'\nTotal Number of High Confidence Samples: {al_model.tot_high_confidence_samples}'
          f'\nTotal Number of Low Confidence Samples: {al_model.tot_low_confidence_samples}',
          file=open(report_path, 'a'))
    return al_model, model_dir


def plot_all(dir_path, result_df):
    print("\nSaving plots...")
    plot_dir = Methods().gen_folder(dir_path, f"plots{datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')}")
    result_df1 = result_df.copy()
    result_df1.drop(result_df1[result_df1.N_Retrains < 2].index, inplace=True)
    result_df1.drop(result_df1[result_df.ML_method.str.contains("CNN_EnS|KNN_EnS|RF_EnS")].index, inplace=True)
    plot_save_results(result_df1, x=result_df1['Number_of_Samples_trained'],
                      y=result_df1['Botnet_label_err_rate_FNR'],
                      hue=result_df1['ML_method'], plot_name="Botnet_Labeling_Error_vs_nSamples",
                      plot_dir=plot_dir)

    plot_save_results(result_df1, x=result_df1['Number_of_Samples_trained'],
                      y=result_df1['Normal_traffic_label_err_rate_FPR'],
                      hue=result_df1['ML_method'], plot_name="Normal_Traffic_Error_vs_nSamples",
                      plot_dir=plot_dir)

    plot_save_results(result_df, x=result_df['N_Retrains'],
                      y=result_df['Queries_to_Oracle'],
                      hue=result_df['ML_method'], plot_name="nQueries_vs_nIterations",
                      plot_dir=plot_dir)

    # plot_save_results(result_df, x=result_df['N_Retrains'],
    #                   y=result_df['F1'],
    #                   hue=result_df['ML_method'], plot_name="F1_vs_nRetrain",
    #                   plot_dir=plot_dir)
    # Prepare new df containing rows with max values of queries
    result_df2 = result_df1[result_df.N_Retrains == result_df.N_Retrains.max()]
    bar_plot_save(result_df2, x=result_df2['ML_method'], y=result_df2['Queries_to_Oracle'], hue=None,
                  plot_name="N_queries_vs_learning algorithm",
                  plot_dir=plot_dir)

    # df4 = pd.DataFrame()
    # df4['ML_method'] = result_df2['ML_method']
    # df4['n_Norm_Traffic_label_err'] = result_df2['n_Norm_Traffic_label_err']
    # df4['n_Bot_label_err'] = result_df2['n_Bot_label_err']
    # # combine 'n_Norm_Traffic_label_err' and 'n_Bot_label_err' to single 'cols' and 'vals' columns
    # dfmelt = df4.melt('ML_method', var_name='cols', value_name='vals')
    # # Exclude the passive learning results
    # dfmelt.drop(dfmelt[dfmelt.ML_method.str.contains('PL_')].index, inplace=True)
    # bar_plot_save(x=dfmelt['ML_method'], y=dfmelt['vals'], hue='cols', df=dfmelt, plot_name='Error_vs_learning algorithm')


def plot_save_results(df, x, y, hue, plot_name, plot_dir):
    sns.lineplot(x=x, y=y, hue=hue, data=df, style=hue, markers=True, dashes=True, err_style="bars", ci=5)
    path = os.path.join(plot_dir, f'{plot_name}.jpg')
    fig = plt.gcf()
    fig.set_size_inches(20, 5)
    fig.savefig(path, dpi=400)
    plt.show()


def bar_plot_save(df, x, y, hue, plot_name, plot_dir):
    sns.barplot(x=x, y=y, hue=hue, data=df)
    path = os.path.join(plot_dir, f'{plot_name}.jpg')
    plt.xticks(rotation=45, horizontalalignment='right', fontweight='light', fontsize='x-large')
    fig = plt.gcf()
    fig.set_size_inches(10, 15)
    fig.savefig(path, dpi=400)
    plt.show()

if __name__ == '__main__':
    main()
