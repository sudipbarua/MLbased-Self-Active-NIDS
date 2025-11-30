from active_self_learner import ActiveSelfLearner
from methods import Methods
import pandas as pd
import time
import pickle

class ActiveLearner(ActiveSelfLearner):
    def __init__(self, model_dir, report_path, al_method=None, batch_size=1000,
                 init_batch_sz=100, active_learner=True, load=False, tune_search_count=20,
                 store_model=True, clusterer_name='HDBSCAN'):
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

    def fit(self, train_df, test_df, tr_set, clf_models=None, cnn_model=None, ml_method=None):
        self.clf_models = clf_models
        self.cnn_model = cnn_model
        self.ml_method = ml_method
        self.tr_set = tr_set
        self.batch_nl = train_df.head(n=self.init_batch_sz)
        # generate/load retrain folder
        if self.load:
            self.retrain_dir = self.model_dir + f'/Retrain_{self.n_retrain}'
        else:
            self.retrain_dir = Methods().gen_folder(self.model_dir, f"Retrain_{self.n_retrain}")
        print(
            "*********************************Fitting model with ML method {}*********************************".format(
                ml_method),
            file=open(self.report_path, 'a'))
        # load the configuration file
        self.param_dict = Methods().parse_config("param_distribution.yaml")
        # label the initial batch manually
        self.batch_nl['new_label'] = self.batch_nl['Label']
        # take validation batch (10% of labelled data pool)
        self.val_batch = self.batch_nl.tail(n=int(len(self.batch_nl) * 0.1))
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
        super().init_train(x_train, y_train)
        # Predict and Label
        while len(train_df.index) > 0:
            self.n_retrain += 1  # iteration count
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
            x = self.batch.drop(['Label'], axis=1)
            self.batch['new_label'] = 0
            self.store_model = False
            if self.active_learner:
                if self.clf_models is not None:
                    if self.al_method == "SVM_C2H":
                        super().get_label_svm_c2h(x)
                    else:
                        super().get_label_clf_us(x)
                elif self.cnn_model is not None:
                    super().get_label_cnn_us(x)
            elif not self.active_learner:
                super().get_label_rs()
            if len(train_df.index) != 0:
                super().get_result()
            else:
                # retrain when all samples are labelled
                self.retrain()
        super().get_final_results()  # create the combined result dataframe
        print(f"*********************************{self.ml_method} Model Fit completed*********************************",
              file=open(self.report_path, 'a'))

    def retrain(self):
        # take validation batch (10% of labelled data pool)
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
            if not self.load:
                self.cnn_model = self.get_clf_tuned(self.cnn_model)  # Tune model
            train_start = time.time()
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
        self.store_model = True
        super().get_result()

