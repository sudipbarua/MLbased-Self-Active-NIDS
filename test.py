import os
import matplotlib.pyplot as plt
import pandas as pd
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
import yaml
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import random
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime
import itertools

def main():
    print("\n Running script...\n")
    # Read data
    # ctu13 = pd.read_csv('./complete_all_scenarios_NoTransfer_withHeader_withLabel-26-1-2021-bak.csv')
    # ctu13 = ctu13.drop(['sTtl'], axis=1)
    # # Convert the ground truth Labels to numeric values
    # ctu13_1 = DataPrep().label_converter(ctu13)
    #
    # print(ctu13_1.groupby('Label').count())


    # m1 = RandomForestClassifier()
    # m2 = KNeighborsClassifier()
    # m3 = LogisticRegression()
    # m4 = CNN.model_cnn()
    # m5 = SVC()
    #
    # print(str(type(m1)).split(".")[-1][:-2], str(type(m2)).split(".")[-1][:-2], str(type(m3)).split(".")[-1][:-2], str(type(m4)).split(".")[-1][:-2] )

    # print("\n\n", str(type(m5)).split(".")[-1][:-2])
    # a_yaml_file = open("param_distribution.yaml")
    # parsed_yaml_file = yaml.load(a_yaml_file, Loader=yaml.FullLoader)
    # with open("param_distribution.yaml", "r") as stream:
    #     try:
    #         parsed_yaml_file = yaml.safe_load(stream)
    #     except yaml.YAMLError as exc:
    #         print(exc)
    # param_dist = parsed_yaml_file['HDBSCAN']
    # param_dicts = []
    # tmp_lst = []
    # for key, lst in param_dist.items():
    #     tmp_lst.append(lst)
    # combo_lst = list(itertools.product(*tmp_lst))
    # for item in combo_lst:
    #     temp_dict = {}
    #     for j, key in zip(item, param_dist):
    #         temp_dict[key] = j
    #     param_dicts.append(temp_dict)
    # print(param_dicts)


    # # print(parsed_yaml_file['CNN_param'])
    #
    # ctu13 = pd.read_csv('./complete_all_scenarios_NoTransfer_withHeader_withLabel-26-1-2021-bak.csv').sample(n=3000)
    # ctu13 = DataPrep().label_converter(ctu13)
    # ctu13 = DataPrep().dataset_cleaning(ctu13)
    # ctu13 = ctu13.drop(['sTtl'], axis=1)
    # val_batch = ctu13.tail(n=int(len(ctu13) * 0.3))
    # ctu13 = ctu13.drop(index=val_batch.index)
    # X = ctu13.drop(['Label'], axis=1)
    # Y = ctu13["Label"]
    #
    # x_test = val_batch.drop(['Label'], axis=1)
    # y_test = val_batch["Label"]
    #
    # param_cnn = parsed_yaml_file['CNN_param']
    # cnn_model = CNN.model_cnn()
    # # cnn_model.get_best_model(val_x=X, val_y=Y, x_test=x_test, y_test=y_test, cnn_params=param_cnn)
    # # Methods().gen_file(cnn_model.model_acc_df,"model_accuracy.csv", path="./results")
    # cnn_model.get_best_model(x_test, y_test, x_test, y_test, param_cnn, 1)
    # cnn_model.cnn_train(X, Y)
    # dir_path = Methods().gen_folder("./results", f"AL_Seq{datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')}")
    # cnn_model.save_cnn_model(dir_path, "CNN")
    # history = cnn_model.history.history
    # df = pd.DataFrame(history)
    # df['epoch'] = np.array(cnn_model.history.epoch)
    # model = KerasClassifier(build_fn=CNN.model_cnn().build_cnn_model(), verbose=0)
    # grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    # grid_result = grid.fit(X, Y)
    # print(grid_result.best_estimator_)
    # print(grid_result.best_score_)

    # clf = RandomForestClassifier(n_estimators=1000)
    # param_rf = parsed_yaml_file['RF_param']
    # search = RandomizedSearchCV(estimator=clf, param_distributions=param_rf, n_jobs=-1)
    # search.fit(X, Y)
    # be = search.best_estimator_

    config_dict = Methods().parse_config("config_directories.yaml")
    # address of main folder
    dir_path = config_dict['main_directory']
    #
    r_df = pd.read_csv('./results/AL_2022-03-01_13-43-41/final_results/AL_result_summary.csv')
    # convert the data types of certain column for further processing
    r_df = r_df.astype({
        'Precision': 'float64',
        'Recall': 'float64',
        'F1': 'float64'
    })

    r_df = r_df.drop(['Training_set'], axis=1)
    r_df_m = pd.melt(r_df, id_vars=['ML_method'], value_vars=['Precision', 'Recall', 'F1'],
                     var_name='score_name', value_name='score')
    # Bar plot
    bar_plot_save(r_df_m, r_df_m['score_name'], r_df_m['score'], r_df_m['ML_method'],
                  "Evaluation_of_different_algorithms", "AL_ML_model", "Score", dir_path)

    # r_df_50 = pd.read_csv('./results/AL_2022-03-01_13-43-41/final_results/results_AL_EnS_avg_th50.csv', index_col=0)
    # r_df_90 = pd.read_csv('./results/AL_2022-03-01_13-43-41/final_results/results_AL_EnS_avg_th90.csv', index_col=0)
    # # r_df_kmeans = pd.read_csv('./results/results_AL_EnS_avg_kmeans.csv', index_col=0)
    #
    # r_df_50['ML_method'] = r_df_50['ML_method'].map({'CNN_EnS_avg': 'CNN_EnS_avg_th_50',
    #                                                  'RF_EnS_avg': 'RF_EnS_avg_th_50',
    #                                                  'KNN_EnS_avg': 'KNN_EnS_avg_th_50',
    #                                                  'EnS_avg': 'EnS_avg_th_50'})
    # r_df_90['ML_method'] = r_df_90['ML_method'].map({'CNN_EnS_avg': 'CNN_EnS_avg_th_90',
    #                                                  'RF_EnS_avg': 'RF_EnS_avg_th_90',
    #                                                  'KNN_EnS_avg': 'KNN_EnS_avg_th_90',
    #                                                  'EnS_avg': 'EnS_avg_th_90'})
    # # r_df_kmeans['ML_method'] = r_df_kmeans['ML_method'].map({'CNN_EnS_avg': 'CNN_EnS_avg_kmeans',
    # #                                                  'RF_EnS_avg': 'RF_EnS_avg_kmeans',
    # #                                                  'KNN_EnS_avg': 'KNN_EnS_avg_kmeans',
    # #                                                  'EnS_avg': 'EnS_avg_kmeans'})
    # r_df_90.to_csv('./results/results_AL_EnS_avg_th_90.csv')
    # r_df_50.to_csv('./results/results_AL_EnS_avg_th_50.csv')
    # # r_df_kmeans.to_csv('./results/results_AL_EnS_avg_th_kmeans.csv')

    print("end")


def bar_plot_save(df, x, y, hue, plot_name, xl, yl, dir_path):
    ax = sns.barplot(x=x, y=y, hue=hue, data=df)
    ax.set(xlabel=xl, ylabel=yl)
    plot_dir = Methods().gen_folder(dir_path, f"plots{datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')}")
    path = os.path.join(plot_dir, f'{plot_name}.jpg')
    plt.xticks(rotation=45, horizontalalignment='right', fontweight='light', fontsize='x-large')
    plt.subplots_adjust(bottom=0.15)
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    fig.savefig(path, dpi=400)
    plt.show()

if __name__ == '__main__':
    main()
