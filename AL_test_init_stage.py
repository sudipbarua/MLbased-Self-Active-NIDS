from data_preparation import DataPrep
from data_splitting import DataSplit
from methods import Methods
import model_fitter
from datetime import datetime
import pandas as pd
from math import comb

def main():
    print("\n Running script...\n")
    grid = {1000: [100, 200, 300, 400, 500],
            3000: [100, 200, 300, 400, 500]}
    # generate main folder
    dir_path = Methods().gen_folder("./results", f"AL_test_init1k{datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')}")
    # generate report.csv file
    report_path = Methods().gen_file(df=pd.DataFrame(list()),
                                     filename=f"report{datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')}.txt",
                                     path=dir_path, dated_sub_dir=False)
    result_df = pd.DataFrame()
    config_dict = Methods().parse_config("config_directories.yaml")
    # get the number of iteration
    n_iter = comb(config_dict['n_chunks'], config_dict['n_chunk_train'])
    for key in grid:
        init_sample_sz = key
        key_path = Methods().gen_folder(dir_path, f"sample_size_{key}")
        # Iterate over different datasets
        for i in range(1, n_iter+1):
            itr_path = Methods().gen_folder(key_path, f"Iteration_{i}")
            itr_path_ds = config_dict[f'Iteration_path_train_{i}']
            print(f"------------------------------Begin Iteration: {i}-------------------------------", file=open(report_path, 'a'))
            print(f"*************************************Dataset**************************************", file=open(report_path, 'a'))
            # load dataset
            for j in range(1, 21):
                sub_sample_path = Methods().gen_folder(itr_path, f"sub_sample_{j}")
                ctu13_train = pd.read_csv(itr_path_ds + f"/train_{i}.csv", index_col=0).sample(init_sample_sz)
                ctu13_test = pd.read_csv(itr_path_ds + f"/test_{i}.csv", index_col=0)
                for val in grid[key]:

                    batch_sz = 3000
                    init_batch_sz = val
                    print(f"-----------------------------Initial batch size {init_batch_sz}--------------------------",
                          file=open(report_path, 'a'))

                    print(f"Train Dataset Dimension: {ctu13_train.shape}", file=open(report_path, 'a'))
                    print(f"Test Dataset Dimension: {ctu13_test.shape}", file=open(report_path, 'a'))
                    print(f"Initial training batch size: {init_batch_sz}", file=open(report_path, 'a'))
                    print(f"Initial training sample size: {init_sample_sz}", file=open(report_path, 'a'))
                    al_model_ens_avg,_ = model_fitter.model_fitter(batch_sz=batch_sz, init_batch_sz=init_batch_sz,
                                                                   learner_typ=True, x=ctu13_train, y=ctu13_test,
                                                                   ml_method='AL_EnS_avg', active_learner='AL_2_stage',
                                                                   tr_set=f'train_{i}_{j}', itr_path=sub_sample_path, report_path=report_path,
                                                                   clusterer_name='HDBSCAN', load=False,
                                                                   init_train_sample_sz=init_sample_sz)

                    print(f"------------------------End iter Initial batch size {init_batch_sz}-----------------------",
                          file=open(report_path, 'a'))

                    result_df = result_df.append(al_model_ens_avg.results_df, ignore_index=True)


            Methods().gen_file(result_df, "results.csv", path=itr_path, dated_sub_dir=False)
            print(f"-------------------------End Iteration: {i}-----------------------------", file=open(report_path, 'a'))


if __name__ == '__main__':
    main()