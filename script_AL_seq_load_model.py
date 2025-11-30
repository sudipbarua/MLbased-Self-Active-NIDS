from datetime import datetime
from active_learner import *
from methods import Methods
import model_fitter


def main():
    print("\n Running script...\n")
    # address of main folder
    dir_path = "./results/AL_Seq_2022-02-06_15-13-12"
    # generate report.csv file
    report_path = Methods().gen_file(df=pd.DataFrame(list()),
                                     filename=f"report{datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')}.csv",
                                     path=dir_path, dated_sub_dir=False)
    batch_sz = 3000
    init_batch_sz = 6000
    result_df = pd.DataFrame()

    # Iterate over different datasets
    for i in range(1,4):
        itr_path = dir_path + f"/Iteration_train_{i}"
        print(f"------------------------------Begin Iteration: train_{i}-------------------------------", file=open(report_path, 'a'))
        print(f"*************************************Dataset**************************************", file=open(report_path, 'a'))
        # Scale dataset
        ctu13_train = pd.read_csv(itr_path+f"/train_{i}.csv", index_col=0)
        ctu13_test = pd.read_csv(itr_path+f"/test_{i}.csv", index_col=0)
        print(f"Train Dataset Dimension: {ctu13_train.shape}", file=open(report_path, 'a'))
        print(f"Test Dataset Dimension: {ctu13_test.shape}", file=open(report_path, 'a'))
        print(f"Initial training batch size: {init_batch_sz}", file=open(report_path, 'a'))
        print(f"Normal training batch size: {batch_sz}", file=open(report_path, 'a'))
        # al_model_rf = model_fitter(batch_sz, init_batch_sz, True, ctu13_train, ctu13_test, "AL_RF_US", tr, itr_path, report_path)
        al_model_ens_avg = model_fitter.model_fitter(batch_sz, init_batch_sz, True, ctu13_train, ctu13_test,
                                        "AL_EnS_avg", f'train_{i}', itr_path, report_path, True)
        # al_model_knn = model_fitter(batch_sz, init_batch_sz, True, ctu13_train, ctu13_test, "AL_KNN_US", tr, itr_path, report_path)
        dal_model_us = model_fitter.model_fitter(batch_sz, init_batch_sz, True, ctu13_train, ctu13_test,
                                    "DAL_US", f'train_{i}', itr_path, report_path, True)
        # al_model_svmC2H = model_fitter(batch_sz, init_batch_sz, True, ctu13_train, ctu13_test, "AL_SVM_C2H", tr, itr_path, report_path)
        # dal_model_tob = model_fitter(batch_sz, init_batch_sz, True, ctu13_train, ctu13_test, "DAL_TOB", tr, itr_path, report_path)
        # pl_model_rf = model_fitter(batch_sz, init_batch_sz, False, ctu13_train, ctu13_test, 'PL_RF', tr, itr_path, report_path)
        # pl_model_knn = model_fitter(batch_sz, init_batch_sz, False, ctu13_train, ctu13_test, 'PL_KNN', tr, itr_path, report_path)
        # dpl_rs_model = model_fitter(batch_sz, init_batch_sz, False, ctu13_train, ctu13_test, "DPL_RS", tr, itr_path, report_path)
        # dpl_tob_model = model_fitter(batch_sz, init_batch_sz, False, ctu13_train, ctu13_test, "DPL_TOB", tr, itr_path, report_path)


        result_df = result_df.append([al_model_ens_avg.results_df, dal_model_us.results_df], ignore_index=True)

        result_path = Methods().gen_file(result_df, f"results{datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')}.csv", path=itr_path, dated_sub_dir=False)
        print(f"-------------------------End Iteration: train_{i}-----------------------------", file=open(report_path, 'a'))

    model_fitter.plot_all(dir_path, result_df)

if __name__ == '__main__':
    main()