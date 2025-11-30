import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import pandas as pd
from data_splitting import *


def plot_save_results(df, x, y, hue, plot_name, xl, yl):
    ax = sns.lineplot(x=x, y=y, hue=hue, data=df, style=hue, markers=True, dashes=True, err_style="bars", ci=5)
    ax.set(xlabel=xl, ylabel=yl)
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)
    path = os.path.join('./fig', '{}_{}.jpg'.format(plot_name, datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    fig = plt.gcf()
    fig.set_size_inches(15, 5)
    fig.savefig(path, dpi=400)
    plt.show()

def bar_plot_save(df, x, y, hue, plot_name, xl, yl):
    ax = sns.barplot(x=x, y=y, hue=hue, data=df)
    ax.set(xlabel=xl, ylabel=yl)
    path = os.path.join('./fig', '{}_{}.jpg'.format(plot_name, datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    plt.xticks(rotation=45, horizontalalignment='right', fontweight='light', fontsize='x-large')
    plt.subplots_adjust(bottom=0.15)
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    fig.savefig(path, dpi=400)
    plt.show()

def comb_col_plot(df, x, y, flds, plot_name, xl, yl):
    # CREATE NEW COLUMN OF CONCATENATED VALUES
    # df['_'.join(flds)] = pd.Series(df.reindex(flds, axis='columns').astype('str').values.tolist()).str.join('_')
    for i in flds:
        df[i] = df[i].astype('str')
    df['_'.join(flds)] = df[flds].agg('_'.join, axis=1)
    # PLOT WITH hue
    ax = sns.lineplot(x=x, y=y, hue='_'.join(flds), data=df, style='_'.join(flds), markers=True, dashes=True, err_style="bars", ci=100)
    ax.set(xlabel=xl, ylabel=yl)
    # ax.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)
    path = os.path.join('./results/AL_test_init1k_2022-02-25_19-38-10/sample_size_3000/', '{}_{}.jpg'.format(plot_name, datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    fig = plt.gcf()
    fig.set_size_inches(20, 5)
    fig.savefig(path, dpi=400)
    plt.show()



# result_df = result_df.astype({
#         'Samples': 'float64',
#         'Norm_tr_label_err_rate': 'float64',
#         'Bot_label_err_rate': 'float64',
#         'F1': 'float64',
#         'N_Retrains': 'float64',
#         'Expert_time': 'float64',
#         'Queries': 'float64',
#         'n_Norm_Traffic_label_err': 'float64',
#         'n_Bot_label_err': 'float64'
#     })

# df1 = pd.read_csv('./results/active_learner_results_AL_PL_SVM.csv')
# df2 = pd.read_csv('./results/active_learner_results_ALPLRFKNN.csv')
# df3 = pd.read_csv('./results/active_learner_results_AL_LR_RF_KNN.csv')
# df4 = pd.read_csv('./results/active_learner_results_ens.csv')
df = pd.read_csv('./results/AL_test_init1k_2022-02-25_19-38-10/sample_size_3000/Iteration_1/results.csv', index_col=0)
# result_df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)
# result_df.drop(result_df[result_df.ML_method.str.contains("EnS_maj")].index, inplace=True)
# df1, df2 = df.copy(), df.copy()
result_df = df.copy()
result_df.drop(result_df[result_df.ML_method.str.contains("CNN|KNN|RF")].index, inplace=True)
# df1.drop(df1[df1.init_batch_sz < 500].index, inplace=True)
# df2.drop(df2[df2.init_batch_sz != 100].index, inplace=True)
# result_df = pd.concat([df1, df2], i1gnore_index=True)
# result_df.drop(result_df[result_df.init_batch_sz != 500].index, inplace=True)
# result_df.drop(result_df[result_df.init_batch_sz != 700].index, inplace=True)
# result_df.drop(result_df[result_df.ML_method.str.contains("EnS|PL_|DAL")].index, inplace=True)
# result_df.drop(result_df[result_df.ML_method.str.contains("EnS|SVM|LR|RF|KNN|PL_")].index, inplace=True)
# gen_file(result_df, "combined_results.csv", path="./results")

# result_df1 = result_df.copy()
# result_df1.drop(result_df1[result_df1.ML_method.str.contains('PL_')].index, inplace=True)
# result_df1.drop(result_df1[result_df1.N_Retrains < 2].index, inplace=True)
# plot_save_results(result_df1, x=result_df1['N_Retrains'],
#                   y=result_df1['Bot_label_err_rate'],
#                   hue=result_df1['ML_method'], plot_name="Botnet_Labeling_Error_vs_nSamples",
#                   xl="No. of Iteration (Retrain)",
#                   yl="Botnet Labeling Error Rate (FNR)"
#                   )

# plot_save_results(result_df1, x=result_df1['N_Retrains'],
#                   y=result_df1['Norm_tr_label_err_rate'],
#                   hue=result_df1['ML_method'], plot_name="Normal_Traffic_Error_vs_nRetrain",
#                   xl="No. of Iteration (Retrain)",
#                   yl="Normal Traffic Labeling Error Rate (FPR)"
#                   )

# plot_save_results(result_df, x=result_df['N_Retrains'],
#                       y=result_df['F1'],
#                       hue=result_df['ML_method'], plot_name="F1_vs_nRetrains",
#                   xl="No. of Iteration (Retrain)",
#                   yl="F1 Score"
#                   )

# plot_save_results(result_df, x=result_df['Expert_time'],
#                       y=result_df['F1'],
#                       hue=result_df['ML_method'], plot_name="F1_vs_ExpertTime",
#                   xl="Annotation Time Taken by Oracle (min)",
#                   yl="F1 Score")
#

plot_save_results(result_df, x=result_df['N_Retrains'],
              y=result_df['Normal_traffic_label_err_rate_FPR'],
              flds=['ML_method', 'init_train_sample_sz', 'init_batch_sz'], plot_name="FPR_vs_n_retrain",
              xl="Number of retrain",
              yl="Normal_traffic_label_err_rate_FPR")

comb_col_plot(result_df, x=result_df['N_Retrains'],
              y=result_df['Botnet_label_err_rate_FNR'],
              flds=['ML_method', 'init_train_sample_sz', 'init_batch_sz'], plot_name="FNR_vs_n_retrain",
              xl="Number of retrain",
              yl="Botnet_label_err_rate_FNR")

#
comb_col_plot(result_df, x=result_df['N_Retrains'],
              y=result_df['Queries_to_Oracle'],
              flds=['ML_method', 'init_train_sample_sz', 'init_batch_sz'], plot_name="Query_vs_n_retrain",
              xl="Number of retrain",
              yl="Queries_to_Oracle")
#
comb_col_plot(result_df, x=result_df['N_Retrains'],
              y=result_df['F1'],
              flds=['ML_method', 'init_train_sample_sz', 'init_batch_sz'], plot_name="F1_vs_n_retrain",
              xl="Number of retrain",
              yl="F1")

result_df['%query'] = result_df['Queries_to_Oracle'] / result_df['Number_of_Samples_trained'] * 100
comb_col_plot(result_df, x=result_df['N_Retrains'],
              y=result_df['%query'],
              flds=['ML_method', 'init_train_sample_sz', 'init_batch_sz'], plot_name="%Query_vs_n_retrain",
              xl="Number of retrain",
              yl='%query')

    # Prepare new df containing rows with max values of queries
# result_df2 = result_df[result_df.N_Retrains == result_df.N_Retrains.max()]
# bar_plot_save(result_df2, x=result_df2['ML_method'], y=result_df2['Queries'], hue=None,
#               plot_name="N_queries_vs_learning algorithm",
#               xl="AL Model",
#               yl="Total Number of Queries")

print("\nDone...")


