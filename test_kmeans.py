import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from unsupervised_labeling import *
from data_preparation import *
import yaml
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import random
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import metrics

def main():
    ctu13 = pd.read_csv('./complete_all_scenarios_NoTransfer_withHeader_withLabel-26-1-2021-bak.csv').sample(n=300)
    ctu13 = DataPrep().label_converter(ctu13)
    ctu13 = DataPrep().dataset_cleaning(ctu13)
    ctu13 = ctu13.drop(['sTtl'], axis=1)
    # UnsupervisedLabeling(kmin=2, kmax=30, sample_sz_ar=[3000, None], model_name='KMeans').elbow_method(ctu13.drop(['Label'], axis=1))
    k_opt = UnsupervisedLabeling(kmin=2, kmax=30, sample_sz_ar=[None], model_name='KMeans').silhouette_method(ctu13.drop(['Label'], axis=1))
    f1 = []
    for k_opt in range(10, 500):
        ctu_nl = UnsupervisedLabeling(model_name='KMeans').get_new_labels(df_scaled=ctu13, n_clusters=k_opt)
        f1.append(metrics.f1_score(ctu_nl['Label'], ctu_nl['new_label']))
    #
    plt.plot(range(10, 500), f1, marker='v')
    plt.title("F1 score vs number of clusters")
    plt.xlabel('Number of clusters')
    plt.ylabel('F1 score')
    plt.show()
    print("\nDone.")
    # ctu_nl = UnsupervisedLabeling().get_new_labels_kmeans(df_scaled=ctu13, n_clusters=k_opt)
    # f1 = metrics.f1_score(ctu_nl['Label'], ctu_nl['new_label'])
    # print(f1)
    # hc = AgglomerativeClustering(linkage='average', affinity='cosine')
    # hc.fit_predict(ctu13.drop(['Label'], axis=1))
    # x = np.resize(hc.labels_, (len(hc.labels_), 1))
    # ctu13['Cluster no.'] = x
    # print(hc.labels_)
    # f1 = metrics.f1_score(ctu13['Label'], ctu13['Cluster no.'])
    # print(f1)


if __name__ == '__main__':
    main()