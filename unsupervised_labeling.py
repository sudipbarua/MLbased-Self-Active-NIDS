"""
This module contains 3 main methods
1. elbow_method - for tuning kmeans
2. silhouette_method -  for tuning kmeans
3. get_new_label - Labels the dataset using clustering method.
"""
from methods import Methods
from model_tuner import ModelTuner
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances_argmin as pda
import timeit
import pandas as pd
from data_preparation import DataPrep
import hdbscan
import warnings
import time
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


def main():
    ctu13 = pd.read_csv('./complete_all_scenarios_NoTransfer_withHeader_withLabel-26-1-2021-bak.csv')
    ctu13_scaled = DataPrep().dataset_cleaning(ctu13.drop('Label', axis=1))
    ul_model = UnsupervisedLabeling(model_name='kmeans', kmax=13,
                                    sample_sz_ar=[3000, 100000, None])
    print("Analyzing cluster size...")
    ul_model.elbow_method(ctu13_scaled)


class UnsupervisedLabeling:

    def __init__(self, model_name, kmin=2, kmax=10, sample_sz_ar=[10000],
                 report_path=None, tune_search=20, loaded_model=None, score_dir=None):
        # min and max values of k for kmeans hyperparameter turning
        self.kmin = kmin
        self.kmax = kmax
        self.sample_sz_ar = sample_sz_ar  # variable sample size for silhouette scoring
        self.model_name = model_name
        self.report_path = report_path
        self.tune_search = tune_search
        self.tune_time = 0
        self.loaded_model = loaded_model
        self.score_dir = score_dir  # dir path for saving scores

    def elbow_method(self, df_scaled):
        # Analyze and plot WCSS and Silhouette Score for different cluster sizes.
        print("\nPerforming Elbow method...")
        # Elbow method
        kmin = self.kmin
        kmax = self.kmax
        sample_sz_ar = self.sample_sz_ar
        wcss = []  # Within Cluster Squared Sum

        for i in range(kmin, kmax + 1):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
            kmeans.fit(df_scaled)
            wcss.append(kmeans.inertia_)

        print("\nPlotting Elbow graph...")
        # Plot Elbow graph
        plt.plot(range(kmin, kmax + 1), wcss, marker='v')
        plt.title("Elbow graph")
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()
        print("\nDone.")

    def silhouette_method(self, df_scaled):
        kmin = self.kmin
        kmax = self.kmax
        k_opt = 2
        # Silhouette Scoring method
        # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
        print("\nPerforming Silhouette Scoring method...\n")
        # loop over different sample sizes
        for sample in self.sample_sz_ar:
            start = timeit.default_timer()
            sil = []
            # min val of k cannot be less than 2
            for k in range(kmin, kmax + 1):
                kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42).fit(df_scaled)
                labels = kmeans.labels_
                sil.append(silhouette_score(df_scaled, labels, metric='mahalanobis', sample_size=sample))
            stop = timeit.default_timer()
            print(f'Execution time of SH with whole dataset : '
                  if sample is None else f'Execution time of SH with {sample} samples: ', stop - start)
            print("\nPlotting...\n")
            k_opt = sil.index(max(sil)) + kmin
            print(f"Optimum Value of K: {k_opt}")
            # Plotting
            plt.plot(range(kmin, kmax + 1), sil,
                     label="Whole dataset" if sample == None else "Sample size ={}".format(sample),
                     marker='v')
            handles, labels = plt.gca().get_legend_handles_labels()
            newLabels, newHandles = [], []
            for handle, label in zip(handles, labels):
                if label not in newLabels:
                    newLabels.append(label)
                    newHandles.append(handle)
            plt.legend(newHandles, newLabels)

        plt.title("Silhouette Score Comparison")
        plt.xlabel('Number of clusters')
        plt.ylabel('silhouette score')
        plt.show()
        return k_opt

    # End of method silhoutte_method

    def fit_kmeans(self):
        if self.n_clusters is None:
            # to get the number of clusters, cluster the dataset with hdbscan
            _,_ = self.get_hdbscan_clusters()
        self.model = KMeans(init="k-means++", n_clusters=self.n_clusters, random_state=42)
        self.model.fit(self.df_1)
        # get indexes of nodes at minimum distances from their clusters centers
        self.min_dist = pda(self.model.cluster_centers_, self.df_1, metric='euclidean')
        # gets the cluster numbers of all of kmeans and converts to df
        x = np.resize(self.model.labels_, (len(self.model.labels_), 1))
        # Add the cluster number to the dataframe
        self.df_scaled['Cluster no.'] = x
        # propagate label of center nodes
        self.df_scaled['new_label'] = self.df_scaled['Label']
        for k in range(self.n_clusters):
            # get the old label of the node nearest to the corresponding center node
            label_center_node = self.df_scaled.iloc[self.min_dist[k], self.df_scaled.columns.get_loc("Label")]
            # fill new_label with label of k where cluster number == k
            self.df_scaled['new_label'] = self.df_scaled['new_label'].mask(self.df_scaled['Cluster no.'] == k,
                                                                           label_center_node)
        self.n_query = self.n_clusters

    def get_hdbscan_clusters(self):
        if self.param is None:
            if self.loaded_model is None:
                # parse parameters and tune
                param_dict = Methods().parse_config("param_distribution.yaml")
                tune_start = time.time()
                # Tune the model
                # search = ModelTuner(model_name='HDBSCAN', n_search=self.tune_search)
                search = ModelTuner(model_name='HDBSCAN', grid_search=True, score_dir=self.score_dir)
                search.random_search(param_dist=param_dict['HDBSCAN'], val_x=self.df_1, save_score=False)
                tune_stop = time.time()
                self.tune_time = tune_stop - tune_start
                self.best_params = search.best_params
                self.model = search.model  # build model
            else:
                self.model = self.loaded_model
        else:
            # Use given parameter to build the model
            self.model = hdbscan.HDBSCAN(**self.param)
        self.model.fit(self.df_1)
        labels = np.resize(self.model.labels_, (len(self.model.labels_), 1))
        prob = np.resize(self.model.probabilities_, (len(self.model.probabilities_), 1))
        self.n_clusters = np.amax(labels) + 1
        return labels, prob


    def fit_hdbscan(self):
        labels, prob = self.get_hdbscan_clusters()
        # Add the cluster number and cluster probabilities to the dataframe
        self.df_scaled['Cluster no.'] = labels
        self.df_scaled['Prob'] = prob
        # extract the outliers and query to oracle
        df_nl = self.df_scaled.loc[self.df_scaled['Cluster no.'] == -1]
        df_nl['new_label'] = df_nl['Label']
        self.n_query = len(df_nl.index)
        self.n_noises = len(df_nl.index)
        self.prob_one = 0
        for k in range(self.n_clusters):
            df_k = self.df_scaled.loc[self.df_scaled['Cluster no.'] == k]  # get kth cluster
            df_k1 = df_k.loc[df_k['Prob'] == 1]  # kth cluster with probability=1
            self.prob_one += len(df_k1.index)
            # #### Contrdict Rule #### Take 10 samples if more instances have probability of 1
            # df_k1 = df_k1.sample(n=10) if len(df_k1) > 10 else df_k1
            # check for inconsistencies in labellings done oracle df_k1['Label']
            if Methods().is_unique(df_k1['Label']):
                # No inconsistency; all labels are either '0' or '1'
                df_k['new_label'] = df_k1['Label'].iloc[0]  # propagate
                self.n_query += len(df_k1['Label'])
            else:
                # inconsistency detected; labels are mixture of '0' and '1'. Propagate all
                maj_elem = Methods().find_majority_elem(df_k1['Label'], th=self.majority_threshold)
                ### Apply majority rule ###
                if maj_elem is None:
                    df_k['new_label'] = df_k['Label']
                    self.n_query += len(df_k['Label'])
                else:
                    df_k['new_label'] = maj_elem
                    self.n_query += len(df_k1['Label'])
            df_nl = df_nl.append(df_k)  # accumulate all clusters
        # self.df_scaled = df_nl.drop(['Prob'], axis=1)
        self.df_scaled = df_nl

    # label the dataset according to k-means clustering
    def get_new_labels(self, df_scaled, n_clusters=None, param=None, majority_threshold=0.5):
        """
        :param df_scaled: dataframe with 'Label'(ground truth) columns
        :param n_clusters: for kmeans only; if the no tuning is required
        :param param: for DBSCAN we can pass the parameters and fit the model without tuning
        :return: df with new labels
        """
        self.n_clusters = n_clusters
        self.df_scaled = df_scaled
        self.param = param
        self.majority_threshold = majority_threshold
        # Drop the ground truth labels
        self.df_1 = df_scaled.drop(['Label'], axis=1)
        # Fit the corresponding model
        if self.model_name == 'kmeans':
            self.fit_kmeans()
        elif self.model_name == 'HDBSCAN':
            self.fit_hdbscan()
        return self.df_scaled
    # End of mehtod get_new_label


if __name__ == '__main__':
    main()
