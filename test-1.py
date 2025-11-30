# from datetime import datetime
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# import pickle
# from joblib import dump, load
# from CNN import CNN
# from sklearn import metrics
# import matplotlib.pyplot as plt

# dir_path = Methods().gen_folder("./results", f"AL_Seq{datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')}")
#     # generate report.csv file
# path = Methods().gen_file(df=pd.DataFrame(list()), filename=f"report{datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')}.csv", path=dir_path, dated_sub_dir=True)
# #     # Read data
# print(dir_path)
# print(path)
#
# file_path = dir_path + "/report.csv"
# print(file_path)

# svc = SVC()
# rf = RandomForestClassifier()
# knn = KNeighborsClassifier()
#
# dump(svc, dir_path+'/svc.joblib')
# dump(rf, dir_path+'/rf.joblib')
# dump(knn, dir_path+'/knn.joblib')
# re = pd.read_csv('./results/AL_Seq_2022-02-03_20-42-05/Iteration_train_3/AL_EnS_avg/Retrain_2/label_predicted_ds_retrain-2.csv')
#
# fpr, tpr, th = metrics.roc_curve(re['Label'], re['p1'], pos_label=1)
# plt.plot(fpr, tpr, marker='.', label='CNN')
# plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
#  # axis labels
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
#  # show the legend
# plt.legend()
#  # show the plot
# plt.show()
# path = "/home/sudip/PycharmProjects/botnet_detection-by_unsupervised_labeling/Unsupervised-Labeling-/results/AL_Seq_2022-02-04_13-43-06/report.csv"
# print("*********************************Model Fit completed*********************************", file=open(path, 'a'))
# from math import comb
# print(comb(5,2))
# from datetime import datetime
# from data_preparation import DataPrep
# from data_splitting import *
import pandas as pd

from methods import Methods
# import pandas as pd
import yaml
#
# dir_path = Methods().gen_folder("./results", f"AL_Seq{datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')}")
# # generate report.csv file
# report_path = Methods().gen_file(df=pd.DataFrame(list()),
#                                  filename=f"report{datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')}.csv",
#                                  path=dir_path, dated_sub_dir=False)
# # Read data
# ctu13 = pd.read_csv('./complete_all_scenarios_NoTransfer_withHeader_withLabel-26-1-2021-bak.csv').sample(n=12000)
# ctu13 = ctu13.drop(['sTtl'], axis=1)
# # Convert the ground truth Labels to numeric values
# ctu13_1 = DataPrep().label_converter(ctu13)
# ds_1 = DataSplit(n_chunks=3,
#                  n_chunk_train=2,
#                  n_chunk_test=1)
# train_df, test_df = ds_1.dataset_spit(ctu13_1, report_path)
#
# batch_sz = 3000
# init_batch_sz = 6000
# result_df = pd.DataFrame()
#
# dict_file = {'main_directory': dir_path,'report_directory' : report_path}
#
#
#
#
# # Iterate over different datasets
# for tr in range(1,4):
#     itr_path = Methods().gen_folder(dir_path, f"Iteration_{tr}")
#     print(f"------------------------------Begin Iteration: {tr}-------------------------------",
#           file=open(report_path, 'a'))
#     print(f"*************************************Dataset**************************************",
#           file=open(report_path, 'a'))
#     dict_file [f"Iteration_{tr}"] = itr_path
#
#
# with open(r'config_directories.yaml', 'w') as file:
#     documents = yaml.dump(dict_file, file)
#
# with open(r'config_directories.yaml') as file:
#     doc = yaml.load(file, Loader=yaml.FullLoader)
#     print(doc['main_directory'])
#     print(doc['report_directory'])
#     for tr in range(1,4):
#         print(doc[f"Iteration_{tr}"])
#
# dict = Methods().parse_config("config_directories.yaml")
# print(dict)
import pandas as pd

config_dict = Methods().parse_config("config_directories.yaml")

foodict = {k: v for k, v in config_dict.items() if k.startswith('results')}

print(foodict)
r_df = pd.DataFrame()
for k, path in config_dict.items():
    if k.startswith('results'):
        temp = pd.read_csv(path, index_col=0)
        r_df = r_df.append(temp, ignore_index=True)

print("end")