from methods import Methods
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from CNN import CNN
from model_tuner import ModelTuner
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime


def main():
    config_dict = Methods().parse_config("config_directories.yaml")
    # address of main folder
    dir_path = config_dict['main_directory']
    results = []
    for i in range(1,4):
        itr_path = config_dict[f'Iteration_path_train_{i}']
        # load dataset
        ctu13_train = pd.read_csv(itr_path + f"/AL_SVM_US/Retrain_11/train_ds_retrain-11.csv", index_col=0)#'.sample(n=4000)
        ctu13_test = pd.read_csv(itr_path + f"/test_{i}.csv", index_col=0)
        x_train = ctu13_train.drop(['Label', 'new_label'], axis=1)
        y_train = ctu13_train['new_label']
        x_test = ctu13_test.drop(['Label'], axis=1)
        y_test = ctu13_test['Label']

        # take validation batch (10% of labelled data pool)
        val_batch = ctu13_train.tail(n=int(len(ctu13_train) * 0.1))
        if len(val_batch) > 6000:
            val_batch = val_batch.sample(n=6000)
        val_x = val_batch.drop(['Label', 'new_label'], axis=1)
        val_y = val_batch["new_label"]
        # load the configuration file
        param_dict = Methods().parse_config("param_distribution.yaml")

        svm_model = SVC()
        svm_model = tune_clf(svm_model, param_dict, val_x, val_y, x_test, y_test)
        svm_model.fit(x_train, y_train)
        pred_svm = svm_model.predict(x_test)
        svm_ev = Methods().get_eval(y_test, pred_svm, 'SVM', f"train_{i}")
        results.append(svm_ev)

        # rf_model = RandomForestClassifier(n_estimators=600, n_jobs=-1)
        # rf_model = tune_clf(rf_model, param_dict, val_x, val_y, x_test, y_test)
        # rf_model.fit(x_train, y_train)
        # pred_rf = rf_model.predict(x_test)
        # rf_ev = Methods().get_eval(y_test, pred_rf, 'Random Forest', f"train_{i}")
        # results.append(rf_ev)

        # knn_model = KNeighborsClassifier(leaf_size=22, n_neighbors=10, p=1, weights='distance',
        #                                  algorithm='ball_tree', n_jobs=-1)
        # knn_model = tune_clf(knn_model, param_dict, val_x, val_y, x_test, y_test)
        # knn_model.fit(x_train, y_train)
        # pred_knn = knn_model.predict(x_test)
        # knn_ev = Methods().get_eval(y_test, pred_knn, 'KNN', f"train_{i}")
        # results.append(knn_ev)

        # param_cnn = param_dict['CNN_param']
        # cnn = CNN.model_cnn()
        # cnn.get_best_model(val_x, val_y, x_test, y_test, param_cnn, n_search=20)
        # cnn.cnn_train(x_train, y_train)
        # pred_cnn = cnn.cnn_test(x_test)
        # cnn_ev = Methods().get_eval(y_test, pred_cnn,'CNN', f"train_{i}")
        # results.append(cnn_ev)

    headers = ['Training_set', 'Feature_set', 'ML_method', 'Precision', 'Recall', 'F1']
    # convert the results to dataframe and
    results_df = pd.DataFrame(np.asarray(results), columns=headers)
    # convert the data types of certain column for further processing
    results_df = results_df.astype({
        'Precision': 'float64',
        'Recall': 'float64',
        'F1': 'float64'
    })
    # save results to .csv format in results folder
    Methods().gen_file(results_df, 'AL_result_summary.csv', dir_path, dated_sub_dir=False)

    r_df = results_df.copy()
    r_df = r_df.drop(['Feature_set', 'Training_set'], axis=1)
    r_df_m = pd.melt(r_df, id_vars=['ML_method'], value_vars=['Precision', 'Recall', 'F1'],
                     var_name='score_name', value_name='score')
    # Bar plot
    bar_plot_save(r_df_m, r_df_m['ML_method'], r_df_m['score'], r_df_m['score_name'],
                  "Evaluation_of_different_algorithms", "AL_ML_model", "Score", dir_path)
    print("\nEnd of execution...\n")


def tune_clf(clf, param_dict, val_x, val_y, x_test, y_test):
    # get the classifier name
    clf_name = str(type(clf)).split(".")[-1][:-2]
    search = ModelTuner(model_name=clf_name, n_search=20)
    search.random_search(param_dist=param_dict[clf_name], val_x=val_x,
                         val_y=val_y, x_test=x_test, y_test=y_test)
    return search.model

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



