"""
***********Important*********
PLease run "script_step_1_initialize.py" file 1st to initialize all the necessary folders
and generate "config_directories.yaml" configuration file
"""
from datetime import datetime

from methods import Methods
import pandas as pd
from math import comb
from model_fitter import model_fitter
import yaml


def main():
    print("\n Running script...\n")
    # al_method_name = "AL_EnS_avg"  # active learner model to be fitted
    # al_method_name = "AL_RF_US"
    al_method_name = "AL_SVM_US"
    # al_method_name = "AL_SVM_C2H"
    # al_method_name = 'PL_RF'
    # al_method_name = 'DAL_US'
    # al_method_name = "DPL_TOB"
    config_dict = Methods().parse_config("config_directories.yaml")
    # address of main folder
    dir_path = config_dict['main_directory']
    # generate report.csv file individual
    report_path = Methods().gen_file(df=pd.DataFrame(list()),
                                     filename=f"report_{al_method_name}_th50{datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')}.txt",
                                     path=dir_path, dated_sub_dir=False)
    # report_path = config_dict['report_directory']  # saves reports in a merged file
    batch_sz = config_dict['batch_sz']
    init_batch_sz = config_dict['init_batch_sz']
    result_df = pd.DataFrame()
    # get the number of iteration
    n_iter = comb(config_dict['n_chunks'], config_dict['n_chunk_train'])
    # Iterate over different datasets
    for i in range(1, n_iter+1):
        itr_path = config_dict[f'Iteration_path_train_{i}']
        print(f"------------------------------Begin Iteration for {al_method_name}: train_{i}-------------------------------",
              file=open(report_path, 'a'))
        # load dataset
        ctu13_train = pd.read_csv(itr_path + f"/train_{i}.csv", index_col=0)#.sample(n=1000)
        ctu13_test = pd.read_csv(itr_path + f"/test_{i}.csv", index_col=0)
        al_model, model_dir = model_fitter(batch_sz=batch_sz,
                                           init_batch_sz=init_batch_sz,
                                           learner_typ=True,
                                           x=ctu13_train, y=ctu13_test,
                                           ml_method=al_method_name,
                                           active_learner=None,
                                           tr_set=f'train_{i}',
                                           itr_path=itr_path,
                                           report_path=report_path,
                                           load=False, clusterer_name='HDBSCAN')
        # al_model_knn = model_fitter(batch_sz, init_batch_sz, True, ctu13_train, ctu13_test, "AL_KNN_US", tr, itr_path, report_path)
        # dal_model_us = model_fitter(batch_sz, init_batch_sz, True, ctu13_train, ctu13_test, "DAL_US", tr, itr_path, report_path)
        # al_model_svmC2H = model_fitter(batch_sz, init_batch_sz, True, ctu13_train, ctu13_test, "AL_SVM_C2H", tr, itr_path, report_path)
        # dal_model_tob = model_fitter(batch_sz, init_batch_sz, True, ctu13_train, ctu13_test, "DAL_TOB", tr, itr_path, report_path)
        # pl_model_rf = model_fitter(batch_sz, init_batch_sz, False, ctu13_train, ctu13_test, 'PL_RF', tr, itr_path, report_path)
        # pl_model_knn = model_fitter(batch_sz, init_batch_sz, False, ctu13_train, ctu13_test, 'PL_KNN', tr, itr_path, report_path)
        # dpl_rs_model = model_fitter(batch_sz, init_batch_sz, False, ctu13_train, ctu13_test, "DPL_RS", tr, itr_path, report_path)
        # dpl_tob_model = model_fitter(batch_sz, init_batch_sz, False, ctu13_train, ctu13_test, "DPL_TOB", tr, itr_path, report_path)

        result_df = result_df.append(al_model.results_df, ignore_index=True)
        result_path = Methods().gen_file(result_df, f"results_{al_method_name}_th50.csv", path=model_dir, dated_sub_dir=False)
        print(f"-------------------------End Iteration: {al_method_name} train_{i}-----------------------------",
              file=open(report_path, 'a'))

    # Parse conmfig file again
    config_dict = Methods().parse_config("config_directories.yaml")
    config_dict[f'results_{al_method_name}_t50'] = result_path  # update/add the path of the corresponding results file

    with open(r'config_directories.yaml', 'w') as file:
        config = yaml.dump(config_dict, file)


if __name__ == '__main__':
    main()