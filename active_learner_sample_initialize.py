from active_self_learner import ActiveSelfLearner
import time
from methods import Methods
import pandas as pd
from sklearn import metrics
import pickle
import numpy as np
from unsupervised_labeling import UnsupervisedLabeling
from data_preparation import DataPrep


class ActiveLearnerSampleInit(ActiveSelfLearner):
    def __init__(self, model_dir, report_path, al_method=None, batch_size=1000,
                 init_batch_sz=100, active_learner=True, load=False, tune_search_count=20,
                 store_model=True, clusterer_name='HDBSCAN', init_train_sample_sz=2000):
        self.model_dir = model_dir
        self.report_path = report_path
        self.al_method = al_method
        self.batch_size = batch_size
        self.init_batch_sz = init_batch_sz
        self.active_learner = active_learner
        self.load = load
        self.tune_search_count = tune_search_count
        self.store_model = store_model
        self.clusterer_name = clusterer_name
        super().__init__(self.model_dir, self.report_path, self.al_method, self.batch_size, self.init_batch_sz,
                         self.active_learner, self.load, self.tune_search_count, self.store_model, self.clusterer_name)
        self.n_retrain = 0
        self.init_train_sample_sz = init_train_sample_sz

    def fit(self, train_df, test_df, tr_set, clf_models=None, cnn_model=None, ml_method=None):
        self.clf_models = clf_models
        self.cnn_model = cnn_model
        self.ml_method = ml_method
        self.tr_set = tr_set
        # generate/load retrain folder
        print("*********************************Fitting model with ML method {}*********************************".format(ml_method),
              file=open(self.report_path, 'a'))
        # load the configuration file
        self.param_dict = Methods().parse_config("param_distribution.yaml")
        ##################### Initial stage ##################
        while len(self.batch_nl.index) < self.init_train_sample_sz:
            self.n_retrain += 1
            # generate/load retrain folder
            if self.load:
                self.retrain_dir = self.model_dir + f'/Init_train_{self.n_retrain}'
            else:
                self.retrain_dir = Methods().gen_folder(self.model_dir, f"Init_train_{self.n_retrain}")
            self.batch = train_df.head(n=self.init_batch_sz)
            train_df = train_df.drop(index=self.batch.index)  # remove batch from the train_df
            self.batch = DataPrep().dataset_cleaning(self.batch)  # scaling
            if self.active_learner==False:
                # label the initial batch manually
                self.batch['new_label'] = self.batch['Label']
                self.batch_nl = self.batch_nl.append(self.batch)
                self.n_query += len(self.batch.index)
            elif self.active_learner==True:
                # Cluster and label propagation
                if self.load:
                    # Load clusterer
                    with open(f'{self.retrain_dir}/HDBSCAN_retrain-{self.n_retrain}_fold-{self.tr_set}.pkl',
                              'rb') as f:
                        clusterer_model = pickle.load(f)
                        self.clusterer = UnsupervisedLabeling(model_name='HDBSCAN', report_path=self.report_path,
                                                              loaded_model=clusterer_model)
                else:
                    self.clusterer = UnsupervisedLabeling(model_name='HDBSCAN', report_path=self.report_path,
                                                          tune_search=self.tune_search_count, loaded_model=None,
                                                          score_dir=self.retrain_dir)
                self.labeled_batch = self.clusterer.get_new_labels(self.batch, majority_threshold=0.9)
                self.batch_nl = self.batch_nl.append(self.labeled_batch.drop(['Cluster no.', 'Prob'], axis=1))
                self.n_clusters = self.clusterer.n_clusters
                self.n_query += self.clusterer.n_query
                self.n_noise = self.clusterer.n_noises
                self.prob_ones = self.clusterer.prob_one
                self.tune_time_clusterer = self.clusterer.tune_time
            self.batch_nl = DataPrep().dataset_cleaning(self.batch_nl)  # scaling
            x_train = self.batch_nl.drop(['Label', 'new_label'], axis=1)
            y_train = self.batch_nl['new_label']
            self.x_test = test_df.drop(['Label'], axis=1)
            self.y_test = test_df['Label']
            self.n_samples += len(self.batch.index)
            # train with initial batch
            self.init_tune = False
            # super().init_train(x_train, y_train)
            ################################
            self.init_train(x_train, y_train)
            ###############################
        self.pred_label_df = pd.DataFrame(columns=self.batch_nl.columns)  # DF containing the predicted labels at each iteration
        ############### Retrain at automation stage ###############
        while len(train_df.index) > 0:
            self.n_retrain += 1
            # generate/load retrain folder
            if self.load:
                self.retrain_dir = self.model_dir + f'/Retrain_{self.n_retrain}'
            else:
                self.retrain_dir = Methods().gen_folder(self.model_dir, f"Retrain_{self.n_retrain}")
            if len(train_df.index) > self.batch_size:
                self.batch = train_df.head(n=self.batch_size)
            else:
                self.batch = train_df
            train_df = train_df.drop(index=self.batch.index)
            self.n_samples += len(self.batch.index)
            self.batch = DataPrep().dataset_cleaning(self.batch)  # scaling
            if self.al_method == "EnS_avg":
                super().retrain_EnS()
            else:
                super().retrain()

        super().get_final_results()
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
                    if clf_name == "model_cnn":
                        train_start = time.time()
                        clf.cnn_train(x_train, y_train)
                        train_stop = time.time()
                        test_start = time.time()
                        self.pred = clf.cnn_test(self.x_test)
                        test_stop = time.time()
                    else:
                        train_start = time.time()
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
            else:
                clf_name = str(type(self.clf_models)).split(".")[-1][:-2]
                train_start = time.time()
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
            train_start = time.time()
            self.cnn_model.cnn_train(x_train, y_train)
            train_stop = time.time()
            # test after initial training
            test_start = time.time()
            self.pred = self.cnn_model.cnn_test(self.x_test)
            test_stop = time.time()
            self.train_time = train_stop - train_start
            self.test_time = test_stop - test_start
            self.get_result()

    # get results at each iteration
    def get_result(self):
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
            fnr = 1 - metrics.recall_score(self.labeled_batch['Label'], self.labeled_batch['new_label'])
            # to calculate, tnr we need to set the positive label to the other class
            fpr = 1 - metrics.recall_score(self.labeled_batch['Label'], self.labeled_batch['new_label'], pos_label=0)
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
