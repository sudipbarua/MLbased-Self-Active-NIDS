from sklearn import metrics
from sklearn.ensemble import VotingClassifier
from sklearn.exceptions import UndefinedMetricWarning
import pickle
from unsupervised_labeling import UnsupervisedLabeling
from model_tuner import ModelTuner
from methods import Methods
import time
import warnings
import pandas as pd
import numpy as np
warnings.simplefilter(action="ignore", category=UndefinedMetricWarning)
from data_preparation import DataPrep


class ActiveSelfLearner:
    def __init__(self, model_dir, report_path, al_method=None, batch_size=500,
                 init_batch_sz=3000, active_learner=True, load=False, tune_search_count=20,
                 store_model=True, clusterer_name='HDBSCAN'):
        """
        :param model_dir: directory path of the model
        :param report_path: path of report.csv file
        :param al_method: "EnS_avg" for Ensemble Sampling by averaging probabilities,
                          "EnS_majority" Ensemble sampling by majority voting,
                          "SVM_C2H" for SVM Nearest to Hyperplane evaluation,
                          "CNN_TOB" for CNN train on batch
                          "CNN_US_CL" for CNN query after clustering (k-means)
        :param batch_size: batch size for retaining
        :param init_batch_sz: initial training dataset size
        :param active_learner: If set to false the model will be trained using passive learning
        :param load: if True, models will be loaded from disk and if False, models will be tuned and
                     saved to disk
        :param tune_search_count: random search count for model hyper-parameter tuning.
        :param clusterer_name: 'HDBSCAN' or 'kmeans'
        """
        self.model_dir = model_dir
        self.report_path = report_path
        self.batch_size = batch_size
        self.init_batch_sz = init_batch_sz
        self.result = []
        self.active_learner = active_learner
        self.al_method = al_method
        self.n_query = 0
        self.n_retrain = 1
        self.n_samples = 0
        self.pred_label_df_tot = pd.DataFrame()  # Dataframe containing all predicted labels
        self.train_time_tot = 0
        self.test_time_tot = 0
        self.tune_time_tot = 0
        self.tune_time_clusterer = 0
        self.clusterer = None
        self.load = load
        self.n_clusters = 0
        self.store_model = store_model
        self.tune_search_count = tune_search_count
        self.clusterer_name = clusterer_name
        self.tr_set = None
        self.batch_nl = pd.DataFrame()
        self.tune_time = 0
        self.n_noise = 0
        self.prob_ones = 0
        self.init_train_sample_sz = 0
        self.n_clusters = 0

    def fit(self, train_df, test_df, tr_set, clf_models=None, cnn_model=None, ml_method=None):
        self.clf_models = clf_models
        self.cnn_model = cnn_model
        self.ml_method = ml_method
        self.tr_set = tr_set
        self.batch_nl = train_df.head(n=self.init_batch_sz)
        # generate/load retrain folder
        if self.load:
            self.retrain_dir = self.model_dir+f'/Retrain_{self.n_retrain}'
        else:
            self.retrain_dir = Methods().gen_folder(self.model_dir, f"Retrain_{self.n_retrain}")
        print("*********************************Fitting model with ML method {}*********************************".format(ml_method),
              file=open(self.report_path, 'a'))
        # load the configuration file
        self.param_dict = Methods().parse_config("param_distribution.yaml")
        # label the initial batch manually
        self.batch_nl['new_label'] = self.batch_nl['Label']
        if self.al_method == "CNN_TOB":
            self.val_batch = self.batch_nl
        else:
            # take validation batch (10% of labelled data pool)
            self.val_batch = self.batch_nl.tail(n=int(len(self.batch_nl)*0.1))
        self.pred_label_df = pd.DataFrame(columns=self.batch_nl.columns)  # DF containing the predicted labels at each iteration
        train_df = train_df.drop(index=self.batch_nl.index)  # remove the initial batch from the train_df
        # self.batch_nl = DataPrep().dataset_cleaning(self.batch_nl)  # scaling
        x_train = self.batch_nl.drop(['Label', 'new_label'], axis=1)
        y_train = self.batch_nl['new_label']
        self.x_test = test_df.drop(['Label'], axis=1)
        self.y_test = test_df['Label']
        self.n_samples += self.init_batch_sz
        # train with initial batch
        self.init_tune = True
        self.init_train(x_train, y_train)
        # Retrain
        while True:
            self.n_retrain += 1
            # generate retrain folder
            self.retrain_dir = Methods().gen_folder(self.model_dir, f"Retrain_{self.n_retrain}")
            if len(train_df.index) > self.batch_size:
                self.batch = train_df.head(n=self.batch_size)
                train_df = train_df.drop(index=self.batch.index)
                self.n_samples += len(self.batch.index)
                # self.batch = DataPrep().dataset_cleaning(self.batch)  # scaling
                if self.al_method == "EnS_avg":
                    self.retrain_EnS()
                elif self.al_method == "EnS_majority":
                    self.retrain_EnS()
                else:
                    self.retrain()
            else:
                self.batch = train_df
                self.n_samples += len(self.batch.index)
                # self.batch = DataPrep().dataset_cleaning(self.batch)  # scaling
                if self.al_method == "EnS_avg":
                    self.retrain_EnS()
                elif self.al_method == "EnS_majority":
                    self.retrain_EnS()
                else:
                    self.retrain()
                break
        self.get_final_results()
        print(f"*********************************{self.ml_method} Model Fit completed*********************************",
              file=open(self.report_path, 'a'))

    # Initial training
    def init_train(self, x_train, y_train):
        if self.clf_models is not None:
            # Ensemble sampling with probability average
            if self.al_method == "EnS_avg":
                train_time_qbc = 0
                for k, clf in self.clf_models.items():
                    # get the classifier name
                    clf_name = str(type(clf)).split(".")[-1][:-2]
                    if self.init_tune and not self.load:
                        clf = self.get_clf_tuned(clf)  # hyperparameter tuning of the classifier if not loaded
                    if clf_name == "model_cnn":
                        train_start = time.time()
                        if self.load:
                            clf.load_cnn_model(self.retrain_dir,
                                               f"/{clf_name}_retrain-{self.n_retrain}_fold-{self.tr_set}")
                        else:
                            clf.cnn_train(x_train, y_train)
                        train_stop = time.time()
                        test_start = time.time()
                        self.pred = clf.cnn_test(self.x_test)
                        test_stop = time.time()
                    else:
                        train_start = time.time()
                        if self.load:
                            # Load classifier
                            with open(f'{self.retrain_dir}/{clf_name}_retrain-{self.n_retrain}_fold-{self.tr_set}.pkl',
                                      'rb') as f:
                                clf = pickle.load(f)
                        else:
                            clf.fit(x_train, y_train)
                        train_stop = time.time()
                        # evaluation after initial training
                        test_start = time.time()
                        self.pred = clf.predict(self.x_test)
                        test_stop = time.time()
                    self.ml_method = k
                    self.train_time = train_stop - train_start
                    self.test_time = test_stop - test_start
                    self.store_model = False  # It is not required to store/save each model individually. All models will be saved at once after fitting all of them
                    self.get_result()
                    train_time_qbc += self.train_time
                    self.clf_models[k] = clf  # Save/overwrite the tuned models
                self.ml_method = self.al_method
                test_start = time.time()
                self.pred = self.pred_label_clf_EnS_avg(self.x_test)
                test_stop = time.time()
                self.train_time = train_time_qbc
                self.test_time = test_stop - test_start
                self.store_model = True  # Save all ensemble models at once.
                self.get_result()
            # Ensemble sampling with majority voting
            elif self.al_method == "EnS_majority":
                self.ens_clf_maj = VotingClassifier(estimators=self.clf_models, voting='soft')
                train_start = time.time()
                self.ens_clf_maj = self.ens_clf_maj.fit(x_train, y_train)
                train_stop = time.time()
                self.ml_method = self.al_method
                test_start = time.time()
                self.pred = self.ens_clf_maj.predict(self.x_test)
                test_stop = time.time()
                self.test_time = test_stop - test_start
                self.train_time = train_stop - train_start
                self.get_result()
                for k, clf in self.ens_clf_maj.named_estimators_.items():
                    test_start = time.time()
                    self.pred = clf.predict(self.x_test)
                    test_stop = time.time()
                    self.ml_method = k
                    self.test_time = test_stop - test_start
                    self.get_result()
            else:
                if self.init_tune and not self.load:
                    # Tune model
                    self.clf_models = self.get_clf_tuned(self.clf_models)
                clf_name = str(type(self.clf_models)).split(".")[-1][:-2]
                train_start = time.time()
                if self.load:
                    # Load classifier
                    with open(f'{self.retrain_dir}/{clf_name}_retrain-{self.n_retrain}_fold-{self.tr_set}.pkl',
                              'rb') as f:
                        self.clf_models = pickle.load(f)
                else:
                    self.clf_models.fit(x_train, y_train)
                train_stop = time.time()
                # test after initial training
                test_start = time.time()
                self.pred = self.clf_models.predict(self.x_test)
                test_stop = time.time()
                self.train_time = train_stop - train_start
                self.test_time = test_stop - test_start
                self.get_result()
        elif self.cnn_model is not None:
            if self.init_tune and not self.load:
                # Tune model
                self.cnn_model = self.get_clf_tuned(self.cnn_model)
            train_start = time.time()
            if self.load:
                self.cnn_model.load_cnn_model(self.retrain_dir,
                                              f"/cnn_model_retrain-{self.n_retrain}_fold-{self.tr_set}")
            else:
                self.cnn_model.cnn_train(x_train, y_train)
            train_stop = time.time()
            # test after initial training
            test_start = time.time()
            self.pred = self.cnn_model.cnn_test(self.x_test)
            test_stop = time.time()
            self.train_time = train_stop - train_start
            self.test_time = test_stop - test_start
            self.get_result()

    # create the combined result dataframe
    def get_final_results(self):
        headers = ['ML_method', 'Training_set', 'N_Retrains', 'Queries_to_Oracle', 'F1',
                   'Normal_traffic_label_err_rate_FPR', 'Botnet_label_err_rate_FNR', 'Number_of_Samples_trained',
                   'Precision', 'Recall', 'Training_Time', 'Testing_Time', 'Tuning_Time', 'Number_of_Clusters',
                   'Cluster Noises', 'Prob_1s', 'init_train_sample_sz', 'init_batch_sz']
        self.results_df = pd.DataFrame(np.asarray(self.result), columns=headers)
        # result df consists of different types of data. But all saved as 'str'. Thus change the floats
        self.results_df = self.results_df.astype({
            'N_Retrains': 'int64',
            'Queries_to_Oracle': 'int64',
            'F1': 'float64',
            'Normal_traffic_label_err_rate_FPR': 'float64',
            'Botnet_label_err_rate_FNR': 'float64',
            'Number_of_Samples_trained': 'float64',
            'Precision': 'float64',
            'Recall': 'float64',
            'Training_Time': 'float64',
            'Testing_Time': 'float64',
            'Tuning_Time': 'float64',
            'Number_of_Clusters': 'int64',
            'Cluster Noises': 'int64',
            'Prob_1s': 'float64',
            'init_train_sample_sz': 'int64',
            'init_batch_sz': 'int64'
        })
        # Calculate the confusion matrix from predicted labels of
        try:
            tn, fp, fn, tp = metrics.confusion_matrix(self.pred_label_df_tot['Label'],
                                                      self.pred_label_df_tot['new_label']).ravel()
        except Exception:
            tn, fp, fn, tp = 0, 0, 0, 0
        self.norm_label_err = fp
        self.bot_label_err = fn
        self.tot_high_confidence_samples = len(self.pred_label_df_tot.index)
        self.tot_low_confidence_samples = self.n_samples - len(self.pred_label_df_tot.index) - self.init_batch_sz

    # Get classifier after hyper-param tuning
    def get_clf_tuned(self, clf):
        tune_start = time.time()
        # get the classifier name
        clf_name = str(type(clf)).split(".")[-1][:-2]
        # validation data
        if len(self.val_batch) > 6000:
            self.val_batch = self.val_batch.sample(n=6000)
        # self.val_batch = DataPrep().dataset_cleaning(self.val_batch)  # scaling
        val_x = self.val_batch.drop(['Label', 'new_label'], axis=1)
        val_y = self.val_batch["new_label"]

        if clf_name == "model_cnn":
            param_cnn = self.param_dict['CNN_param']
            clf.get_best_model(val_x, val_y, self.x_test, self.y_test, param_cnn, n_search=self.tune_search_count)
            tune_stop = time.time()
            self.tune_time = tune_stop - tune_start
            print(f"Best parameters set for {clf_name}:\n{clf.best_params}"
                  f"\nValidation Data Size: {self.val_batch.shape}"
                  f"\n----------------------------------------------------------------------------------",
                  file=open(self.report_path, 'a'))
            return clf
        else:
            search = ModelTuner(model_name=clf_name, n_search=self.tune_search_count)
            search.random_search(param_dist=self.param_dict[clf_name], val_x=val_x,
                                 val_y=val_y, x_test=self.x_test, y_test=self.y_test)
            tune_stop = time.time()
            self.tune_time = tune_stop - tune_start
            print(f"Best parameters set for {clf_name}:\n{search.best_params}"
                  f"\nValidation Batch Size: {self.val_batch.shape}"
                  f"\n----------------------------------------------------------------------------------",
                  file=open(self.report_path, 'a'))
            return search.model

    # Retraining method for Query By Committee labeling
    def retrain_EnS(self):
        x = self.batch.drop(['Label'], axis=1)
        self.batch['new_label'] = 0
        if self.al_method == "EnS_avg":
            self.get_label_clf_EnS_avg(x)
        elif self.al_method == "EnS_majority":
            self.get_label_clf_us(x)
        # take validation batch (10% of labelled data pool)
        self.val_batch = self.batch_nl.tail(n=int(len(self.batch_nl) * 0.1))
        # self.batch_nl = DataPrep().dataset_cleaning(self.batch_nl)  # scaling
        x_train = self.batch_nl.drop(['Label', 'new_label'], axis=1)
        y_train = self.batch_nl['new_label']
        if self.al_method == "EnS_avg":
            # Get individual predictions and results for different classifiers
            train_time_qbc = 0
            for k, clf in self.clf_models.items():
                # get the classifier name
                clf_name = str(type(clf)).split(".")[-1][:-2]
                if not self.load:
                    clf = self.get_clf_tuned(clf)  # hyperparameter tuning of the classifier
                if clf_name == "model_cnn":
                    train_start = time.time()
                    if self.load:
                        clf.load_cnn_model(self.retrain_dir,
                                           f"/{clf_name}_retrain-{self.n_retrain}_fold-{self.tr_set}")
                    else:
                        clf.cnn_train(x_train, y_train, epoch=clf.best_params['epochs'], batch_size=clf.best_params['batch_size'])
                    train_stop = time.time()
                    test_start = time.time()
                    self.pred = clf.cnn_test(self.x_test)
                    test_stop = time.time()
                else:
                    train_start = time.time()
                    if self.load:
                        # Load classifier
                        with open(f'{self.retrain_dir}/{clf_name}_retrain-{self.n_retrain}_fold-{self.tr_set}.pkl',
                                  'rb') as f:
                            clf = pickle.load(f)
                    else:
                        clf.fit(x_train, y_train)
                    train_stop = time.time()
                    test_start = time.time()
                    self.pred = clf.predict(self.x_test)
                    test_stop = time.time()
                self.clf_models[k] = clf  # Save/overwrite the tuned models
                self.ml_method = k
                self.train_time = train_stop - train_start
                train_time_qbc += self.train_time
                self.test_time = test_stop - test_start
                self.store_model = False  # It is not required to store/save each model individually. All models will be saved at once after fitting all of them
                self.get_result()  # get results of individual classifiers
            self.ml_method = self.al_method
            test_start = time.time()
            self.pred = self.pred_label_clf_EnS_avg(self.x_test)
            test_stop = time.time()
            self.test_time = test_stop - test_start
            self.train_time = train_time_qbc
            self.store_model = True  # Save all ensemble models at once.
            self.get_result()  # get results of the committee of the classifiers
        elif self.al_method == "EnS_majority":
            train_start = time.time()
            self.ens_clf_maj = self.ens_clf_maj.fit(x_train, y_train)
            train_stop = time.time()
            self.ml_method = self.al_method
            test_start = time.time()
            self.pred = self.ens_clf_maj.predict(self.x_test)
            test_stop = time.time()
            self.train_time = train_stop - train_start
            self.test_time = test_stop - test_start
            self.get_result()  # get results of the committee of the classifiers
            for k, clf in self.ens_clf_maj.named_estimators_.items():
                test_start = time.time()
                self.pred = clf.predict(self.x_test)
                test_stop = time.time()
                self.ml_method = k
                self.test_time = test_stop - test_start
                self.get_result()  # get results of individual classifiers

    def retrain(self):
        x = self.batch.drop(['Label'], axis=1)
        self.batch['new_label'] = 0
        if self.active_learner:
            if self.clf_models is not None:
                if self.al_method == "SVM_C2H":
                    self.get_label_svm_c2h(x)
                else:
                    self.get_label_clf_us(x)
            elif self.cnn_model is not None:
                self.get_label_cnn_us(x)
        elif not self.active_learner:
            self.get_label_rs()
        # take validation batch (10% of labelled data pool)
        if self.al_method == "CNN_TOB":
            # self.val_batch = self.batch_nl
            pass
        else:
            self.val_batch = self.batch_nl.tail(n=int(len(self.batch_nl) * 0.1))
        # self.batch_nl = DataPrep().dataset_cleaning(self.batch_nl)  # scaling
        x_train = self.batch_nl.drop(['Label', 'new_label'], axis=1)
        y_train = self.batch_nl['new_label']
        train_stop, train_start, test_stop, test_start = 0, 0, 0, 0
        if self.clf_models is not None:
            clf_name = str(type(self.clf_models)).split(".")[-1][:-2]  # get classifier name
            if not self.load:
                # Tune model
                self.clf_models = self.get_clf_tuned(self.clf_models)
            train_start = time.time()
            if self.load:
                # Load classifier
                with open(f'{self.retrain_dir}/{clf_name}_retrain-{self.n_retrain}_fold-{self.tr_set}.pkl',
                          'rb') as f:
                    self.clf_models = pickle.load(f)
            else:
                self.clf_models.fit(x_train, y_train)  # retrain
            train_stop = time.time()
            test_start = time.time()
            self.pred = self.clf_models.predict(self.x_test)
            test_stop = time.time()
        elif self.cnn_model is not None:
            if self.al_method == "CNN_TOB":
                pass  # No tuning required for batch training
            else:
                # if not train on batch then do tuning
                if not self.load:
                    self.cnn_model = self.get_clf_tuned(self.cnn_model)  # Tune model
            train_start = time.time()
            if self.al_method == "CNN_TOB":
                self.cnn_model.cnn_train_on_batch(x_train, y_train)
            else:
                if self.load:
                    self.cnn_model.load_cnn_model(self.retrain_dir,
                                                  f"/cnn_model_retrain-{self.n_retrain}_fold-{self.tr_set}")
                else:
                    self.cnn_model.cnn_train(x_train, y_train)
            train_stop = time.time()
            test_start = time.time()
            self.pred = self.cnn_model.cnn_test(self.x_test)
            test_stop = time.time()
        self.train_time = train_stop - train_start
        self.test_time = test_stop - test_start
        self.get_result()

    # Labeling by evaluating data-points closest to hyperplane
    def get_label_svm_c2h(self, x):
        d = self.clf_models.decision_function(x)
        self.batch['dist'] = d
        self.pred_label_df = pd.DataFrame(columns=self.batch.columns)
        for i, row in self.batch.iterrows():
            if row['dist'] > 1.0:
                self.batch.loc[i, 'new_label'] = 1
                # Accumulate the classifier labelled (predicted) data points
                self.pred_label_df.loc[i] = self.batch.loc[i]
            elif row['dist'] < -1.0:
                self.batch.loc[i, 'new_label'] = 0
                # Accumulate the classifier labelled (predicted) data points
                self.pred_label_df.loc[i] = self.batch.loc[i]
            else:
                # query to the Oracle
                self.n_query += 1
                self.batch.loc[i, 'new_label'] = self.batch.loc[i, 'Label']
        self.batch_nl = self.batch_nl.append(self.batch.drop(['dist'], axis=1))
        self.pred_label_df_tot = self.pred_label_df_tot.append(self.pred_label_df)

    # Labeling method with random sampling for passive learning
    # N.B. if al_method is "CNN_TOB", it takes only the current batch for CNN train
    def get_label_rs(self):
        for i, row in self.batch.iterrows():
            # query to the Oracle
            self.n_query += 1
            self.batch.loc[i, 'new_label'] = self.batch.loc[i, 'Label']
        if self.al_method == "CNN_TOB":
            self.batch_nl = self.batch
        else:
            self.batch_nl = self.batch_nl.append(self.batch)

    # Predicts label for pred_label_clf_EnS_avg() and get_label_clf_EnS_avg() methods
    def get_prob_avg_EnS(self, x):
        T = np.zeros((len(x.index), 1))  # Initialize with a column of zeros
        for k, clf in self.clf_models.items():
            # get the classifier name
            clf_name = str(type(self.clf_models[k])).split(".")[-1][:-2]
            if clf_name == "model_cnn":
                p1 = clf.cnn_proba(x)
                p0 = 1 - p1
                T = np.concatenate((T, p0, p1), axis=1)
            else:
                p = clf.predict_proba(x)
                T = np.append(T, p, axis=1)
        T = pd.DataFrame(np.delete(T, 0, axis=1))  # Delete the initial column no: 0
        self.prob_avg = T.copy()
        mask = np.arange(len(T.columns)) % 2
        self.prob_avg['p0'] = T.iloc[:, mask == 0].mean(axis=1)
        self.prob_avg['p1'] = T.iloc[:, mask == 1].mean(axis=1)

    # Predict labels with ensemble sampling by averaging
    def pred_label_clf_EnS_avg(self, x_test):
        x = x_test.copy()
        self.get_prob_avg_EnS(x)
        x[['p0', 'p1']] = self.prob_avg[['p0', 'p1']].to_numpy()
        for i, row in x.iterrows():
            if row['p0'] > 0.5:
                x.loc[i, 'new_label'] = 0
            elif row['p1'] > 0.5:
                x.loc[i, 'new_label'] = 1
        return x['new_label']

    # Labeling by Ensemble sampling (average) Query by Committee plus clustering method
    def get_label_clf_EnS_avg(self, x):
        cb = pd.DataFrame()  # initialize the df for uncertain labels
        self.get_prob_avg_EnS(x)
        self.batch[['p0', 'p1']] = self.prob_avg[['p0', 'p1']].to_numpy()
        self.pred_label_df = pd.DataFrame(columns=self.batch.columns)
        for i, row in self.batch.iterrows():
            if row['p0'] > 0.7:
                self.batch.loc[i, 'new_label'] = 0
                # Accumulate the classifier labelled (predicted) data points
                self.pred_label_df.loc[i] = self.batch.loc[i]
            elif row['p1'] > 0.7:
                self.batch.loc[i, 'new_label'] = 1
                # Accumulate the classifier labelled (predicted) data points
                self.pred_label_df.loc[i] = self.batch.loc[i]
            else:
                # query to the Oracle
                self.batch = self.batch.drop(i, axis=0)
                cb = cb.append(row)  # accumulate the queries to oracle
        cb = cb.drop(['p0', 'p1', 'new_label'], axis=1)  # Drop the unnecessary columns 'p0', 'p1'
        # Cluster and label propagation
        if self.load:
            # Load clusterer
            with open(f'{self.retrain_dir}/HDBSCAN_retrain-{self.n_retrain}_fold-{self.tr_set}.pkl',
                      'rb') as f:
                clusterer_model = pickle.load(f)
                self.clusterer = UnsupervisedLabeling(model_name=self.clusterer_name, report_path=self.report_path,
                                                      loaded_model=clusterer_model)
        else:
            self.clusterer = UnsupervisedLabeling(model_name=self.clusterer_name, report_path=self.report_path,
                                                  tune_search=self.tune_search_count, loaded_model=None)
        # cb = DataPrep().dataset_cleaning(cb)  # scaling
        cb_nl = self.clusterer.get_new_labels(df_scaled=cb, n_clusters=10)
        # save the sample having low confidence along with the labels annotated by oracle
        Methods().gen_file(cb_nl, f"low_confidence_samples_retrain-{self.n_retrain}.csv", self.retrain_dir, False)
        self.n_clusters = self.clusterer.n_clusters
        self.tune_time_clusterer = self.clusterer.tune_time
        self.n_query += self.clusterer.n_query
        # add to the labelled data pool
        if self.clusterer_name == 'HDBSCAN':
            cb_nl = cb_nl.drop(['Cluster no.', 'Prob'], axis=1)
            print(f"Best parameters set for HDBSCAN:\n{self.clusterer.best_params}"
                  f"\n-------------------------------------------------------------------------------------",
                  file=open(self.report_path, 'a')) if not self.load else print("Clusterred using loaded model")
        elif self.clusterer_name == 'kmeans':
            cb_nl = cb_nl.drop(['Cluster no.'], axis=1)
        self.batch_nl = self.batch_nl.append([self.batch.drop(['p0', 'p1'], axis=1), cb_nl])
        self.pred_label_df_tot = self.pred_label_df_tot.append(self.pred_label_df)

    # Labeling method for uncertainty sampling with corresponding classifier model
    def get_label_clf_us(self, x):
        if self.al_method == "EnS_majority":
            T = self.ens_clf_maj.predict_proba(x)
        else:
            T = self.clf_models.predict_proba(x)
        self.batch['p0'] = T[:, 0]
        self.batch['p1'] = T[:, 1]
        self.pred_label_df = pd.DataFrame(columns=self.batch.columns)
        for i, row in self.batch.iterrows():
            if row['p0'] > 0.7:
                self.batch.loc[i, 'new_label'] = 0
                # Accumulate the classifier labelled (predicted) data points
                self.pred_label_df.loc[i] = self.batch.loc[i]
            elif row['p1'] > 0.7:
                self.batch.loc[i, 'new_label'] = 1
                # Accumulate the classifier labelled (predicted) data points
                self.pred_label_df.loc[i] = self.batch.loc[i]
            else:
                # query to the Oracle
                self.n_query += 1
                self.batch.loc[i, 'new_label'] = self.batch.loc[i, 'Label']
        self.batch_nl = self.batch_nl.append(self.batch.drop(['p0', 'p1'], axis=1))
        self.pred_label_df_tot = self.pred_label_df_tot.append(self.pred_label_df)

    # Labeling method for uncertainty sampling with corresponding CNN model
    # N.B. if al_method == "CNN_TOB", it takes only the current batch, not appending to the labeled data pool
    def get_label_cnn_us(self, x):
        if self.al_method == "CNN_US_CL":
            cb = pd.DataFrame()  # initialize the df for uncertain labels
        else:
            pass
        T = self.cnn_model.cnn_proba(x)
        self.batch['p1'] = T
        self.pred_label_df = pd.DataFrame(columns=self.batch.columns)
        for i, row in self.batch.iterrows():
            if row['p1'] > 0.7:
                self.batch.loc[i, 'new_label'] = 1
                # Accumulate the classifier labelled (predicted) data points
                self.pred_label_df.loc[i] = self.batch.loc[i]
            elif row['p1'] < 0.3:
                self.batch.loc[i, 'new_label'] = 0
                # Accumulate the classifier labelled (predicted) data points
                self.pred_label_df.loc[i] = self.batch.loc[i]
            else:
                # query to the Oracle
                if self.al_method == "CNN_US_CL":
                    self.batch = self.batch.drop(i, axis=0)
                    cb = cb.append(row)  # accumulate the queries to oracle
                else:
                    self.n_query += 1
                    self.batch.loc[i, 'new_label'] = self.batch.loc[i, 'Label']
        if self.al_method == "CNN_TOB":
            self.batch_nl = self.batch.drop(['p1'], axis=1)
        elif self.al_method == "CNN_US_CL":
            n_query = 10
            cb = cb.drop(['p1'], axis=1)
            # Cluster and label propagation
            cb_nl = UnsupervisedLabeling(model_name='KMeans').get_new_labels(df_scaled=cb, n_clusters=n_query)
            self.n_query += n_query
            # add to the labelled data pool
            self.batch_nl = self.batch_nl.append([self.batch.drop(['p1'], axis=1), cb_nl.drop(['Cluster no.'], axis=1)])
        else:
            self.batch_nl = self.batch_nl.append(self.batch.drop(['p1'], axis=1))
        self.pred_label_df_tot = self.pred_label_df_tot.append(self.pred_label_df)

    # get results at each iteration
    def get_result(self):
        # save train, validation and predicted_labeled datasets at retrain_dir path
        Methods().gen_file(self.batch_nl, f"train_ds_retrain-{self.n_retrain}.csv", self.retrain_dir, False)
        try:
            Methods().gen_file(self.pred_label_df, f"label_predicted_ds_retrain-{self.n_retrain}.csv", self.retrain_dir, False)
        except:
            pass
        Methods().gen_file(pd.DataFrame(self.pred), f"test_data_predicted_labels_by_trained_model_retrain-{self.n_retrain}.csv",
                           self.retrain_dir,
                           False)
        if not self.load and self.store_model:
            # loading not true, save the model
            self.save_models()
        try:
            # Try Updating the scores
            self.f1 = metrics.f1_score(self.y_test, self.pred)
            self.pr = metrics.precision_score(self.y_test, self.pred)
            self.recall = metrics.recall_score(self.y_test, self.pred)
        except:
            # keep the scores from previous iteration
            self.f1 = self.f1
            self.pr = self.pr
            self.recall = self.recall
        try:
            fnr = 1 - metrics.recall_score(self.pred_label_df['Label'], self.pred_label_df['new_label'])
            # to calculate, tnr we need to set the positive label to the other class
            fpr = 1 - metrics.recall_score(self.pred_label_df['Label'], self.pred_label_df['new_label'], pos_label=0)
        except:
            fnr, fpr = 0, 0
        r = [self.ml_method, self.tr_set, self.n_retrain, self.n_query, self.f1, fpr, fnr,
             self.n_samples, self.pr, self.recall, self.train_time, self.test_time, self.tune_time, self.n_clusters,
             self.n_noise, self.prob_ones, self.init_train_sample_sz, self.init_batch_sz]
        self.result.append(r)
        self.train_time_tot += self.train_time
        self.test_time_tot += self.test_time
        self.tune_time_tot += self.tune_time + self.tune_time_clusterer
        print(f'{self.ml_method} Model \nNo. of Retrains: {self.n_retrain} \nNo. of Queries: {self.n_query}'
              f'\nNo of Samples: {self.n_samples} \nF1 Score: {self.f1} \nPrecision: {self.pr}'
              f'\nRecall: {self.recall} \nTotal Training Time: {self.train_time_tot}'
              f'\nTotal Tune Time: {self.tune_time_tot} \nTotal Test Time: {self.test_time_tot}', file=open(self.report_path, 'a'))
        print("---------------------------------------------------------------------------------------",
              file=open(self.report_path, 'a'))

    # Save the models at each iteration
    def save_models(self):
        if self.clf_models is not None:
            if self.al_method == "EnS_avg":
                for k, clf in self.clf_models.items():
                    # get the classifier name
                    clf_name = str(type(self.clf_models[k])).split(".")[-1][:-2]
                    if clf_name == "model_cnn":
                        # save the classifier to retrain directory
                        self.clf_models[k].save_cnn_model(self.retrain_dir, f"/{clf_name}_retrain-{self.n_retrain}_fold-{self.tr_set}")
                        self.clf_models[k].plot_loss_validation(
                            self.retrain_dir + f"/epoch_vs_loss_retrain-{self.n_retrain}_fold-{self.tr_set}.jpg")
                        self.clf_models[k].plot_accuracy(
                            self.retrain_dir + f"/accuracy_vs_loss_retrain-{self.n_retrain}_fold-{self.tr_set}.jpg")
                        history_df = pd.DataFrame(self.clf_models[k].history.history)
                        history_df['epoch'] = np.array(self.clf_models[k].history.epoch)
                        Methods().gen_file(history_df, f"cnn_history_retrain-{self.n_retrain}_fold-{self.tr_set}.csv",
                                           self.retrain_dir, dated_sub_dir=False)
                    else:
                        # save the classifier to retrain directory
                        with open(f'{self.retrain_dir}/{clf_name}_retrain-{self.n_retrain}_fold-{self.tr_set}.pkl',
                                  'wb') as f:
                            pickle.dump(self.clf_models[k], f)
                        # dump(self.clf_models[k], self.retrain_dir + f"/{clf_name}_retrain-{self.n_retrain}_fold-{self.tr_set}.joblib")
            else:
                # get the classifier name
                clf_name = str(type(self.clf_models)).split(".")[-1][:-2]
                # save the classifier to retrain directory
                with open(f'{self.retrain_dir}/{clf_name}_retrain-{self.n_retrain}_fold-{self.tr_set}.pkl', 'wb') as f:
                    pickle.dump(self.clf_models, f)
                # dump(self.clf_models, self.retrain_dir + f"/{clf_name}_retrain-{self.n_retrain}_fold-{self.tr_set}.joblib")
        elif self.cnn_model is not None:
            # save the classifier to retrain directory
            self.cnn_model.save_cnn_model(self.retrain_dir, f"/cnn_model_retrain-{self.n_retrain}_fold-{self.tr_set}")
            self.cnn_model.plot_loss_validation(self.retrain_dir+f"/epoch_vs_loss_retrain-{self.n_retrain}_fold-{self.tr_set}.jpg")
            self.cnn_model.plot_accuracy(self.retrain_dir+f"/accuracy_vs_loss_retrain-{self.n_retrain}_fold-{self.tr_set}.jpg")
            history_df = pd.DataFrame(self.cnn_model.history.history)
            history_df['epoch'] = np.array(self.cnn_model.history.epoch)
            Methods().gen_file(history_df, f"cnn_history_retrain-{self.n_retrain}_fold-{self.tr_set}.csv",
                               self.retrain_dir, dated_sub_dir=False)
        if self.clusterer is not None:
            # save the clusterer
            with open(f'{self.retrain_dir}/HDBSCAN_retrain-{self.n_retrain}_fold-{self.tr_set}.pkl', 'wb') as f:
                pickle.dump(self.clusterer.model, f)

