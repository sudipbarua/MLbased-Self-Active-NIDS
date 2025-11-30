import errno
import os
from datetime import datetime
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

from unsupervised_labeling import *
from data_preparation import *
from data_splitting import DataSplit
from feature_selection import *
from methods import Methods


def main():
    print("\n Running script...\n")
    # Read data
    ctu13 = pd.read_csv('./complete_all_scenarios_NoTransfer_withHeader_withLabel-26-1-2021-bak.csv')
    ctu13 = ctu13.drop(['sTtl'], axis=1)
    # Convert the ground truth Labels to numeric values
    ctu13_1 = DataPrep().label_converter(ctu13)
    # Data splitting
    ds_1 = DataSplit(n_chunks=5,
                     n_chunk_train=3,
                     n_chunk_test=2)
    train_df, test_df = ds_1.dataset_spit(ctu13_1, save_file=False)
    # SVM model
    svm_model = SVC()
    # Random forest model
    rf_model = RandomForestClassifier(n_estimators=1000)
    # Iterate over different datasets
    results = []
    for tr, ts in zip(train_df, test_df):
        # Scale dataset
        ctu13_train = DataPrep().dataset_cleaning(train_df[tr])
        ctu13_test = DataPrep().dataset_cleaning(test_df[ts])
        # feature selection: Backward elimination
        # new_feat = FeatureSelection().backward_elimination(ctu13_train)
        # feature selection with Pearson's correlation method
        new_feat = FeatureSelection().correl_selection(ctu13_train)
        feat_set = " & ".join(new_feat)
        ctu13_train = FeatureSelection().get_new_feat_df(new_feat, ctu13_train)
        ctu13_test = FeatureSelection().get_new_feat_df(new_feat, ctu13_test)
        # Iterate over different cluster sizes
        for i in range(3, 13):
            # New label by K-Means for train and test set
            ctu13_tr_nl = UnsupervisedLabeling(model_name='KMeans').get_new_labels(ctu13_train, i)
            # Pre evaluation of new labels of train set
            pre_ev = Methods().get_eval(ctu13_tr_nl['Label'], ctu13_tr_nl['new_label'], i, 'Pre-Evaluation', tr, feature_set=feat_set)
            results.append(pre_ev)

            # Training and testing
            x_train = ctu13_tr_nl.drop(['Label', 'Cluster no.', 'new_label'], axis=1)
            y_train = ctu13_tr_nl['new_label']
            x_test = ctu13_test.drop(['Label'], axis=1)
            y_test = ctu13_test['Label']
            # save files
            # gen_file(x_train, "x_{}_n-clusters-{}.csv".format(tr, i), './train_test_df')
            # gen_file(y_train, "y_{}_n-clusters-{}.csv".format(tr, i), './train_test_df')
            # gen_file(x_test, "x_{}_n-clusters-{}.csv".format(ts, i), './train_test_df')
            # gen_file(y_test, "y_{}_n-clusters-{}.csv".format(ts, i), './train_test_df')

            svm_model.fit(x_train, y_train)
            pred_svm = svm_model.predict(x_test)
            svm_ev = Methods().get_eval(y_test, pred_svm, i, 'SVM', tr, feat_set)
            results.append(svm_ev)

            rf_model.fit(x_train, y_train)
            pred_rf = rf_model.predict(x_test)
            rf_ev = Methods().get_eval(y_test, pred_rf, i, 'Random Forest', tr, feat_set)
            results.append(rf_ev)

    headers = ['Training_set', 'Feature_set', 'n_clusters', 'ML_method', 'Precision', 'Recall', 'F1']
    # convert the results to dataframe and
    results_df = pd.DataFrame(np.asarray(results), columns=headers)
    # save results to .csv format in results folder
    Methods().gen_file(results_df, 'Result_summary.csv', './results')
    # convert the data types of certain column for further processing
    results_df = results_df.astype({
        'n_clusters': 'int64',
        'Precision': 'float64',
        'Recall': 'float64',
        'F1': 'float64'
    })
    # Bar plot
    Methods().plot_save_scores(results_df, 'F1')
    Methods().plot_save_scores(results_df, 'Precision')
    Methods().plot_save_scores(results_df, 'Recall')
    print("\nEnd of execution...\n")


if __name__ == '__main__':
    main()
