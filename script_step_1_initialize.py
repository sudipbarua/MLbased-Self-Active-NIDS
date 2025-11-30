from methods import Methods
from datetime import datetime
import pandas as pd
from data_preparation import DataPrep
from data_splitting import DataSplit
import yaml

def main():
    print("\n Running script...\n")
    # generate main folder
    print("Creating main folder...")
    dir_path = Methods().gen_folder("./results", f"AL{datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')}")
    # generate report.csv file
    report_path = Methods().gen_file(df=pd.DataFrame(list()),
                                     filename=f"report{datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')}.txt",
                                     path=dir_path, dated_sub_dir=False)
    # Read data
    ds1 = pd.read_csv('./SDN_DS/Friday-WorkingHours-Morning_clean.csv', index_col=0)
    ds2 = pd.read_csv('./SDN_DS/Friday-WorkingHours-Afternoon-DDos_clean.csv', index_col=0)
    ds3 = pd.read_csv('./SDN_DS/Friday-WorkingHours-Afternoon-PortScan_clean.csv', index_col=0)
    ds4 = pd.read_csv('./SDN_DS/Monday-WorkingHours_clean.csv', index_col=0)
    ds5 = pd.read_csv('./SDN_DS/Thursday-WorkingHours-Afternoon-Infilteration_clean.csv', index_col=0)
    ds6 = pd.read_csv('./SDN_DS/Thursday-WorkingHours-Morning-WebAttacks_clean.csv', index_col=0)
    ds7 = pd.read_csv('./SDN_DS/Tuesday-WorkingHours_clean.csv', index_col=0)
    ds8 = pd.read_csv('./SDN_DS/Wednesday-workingHours_clean.csv', index_col=0)
    # combine all DS
    # ds = pd.read_csv('./complete_all_scenarios_NoTransfer_withHeader_withLabel-26-1-2021-bak.csv').sample(n=160000)
    ds = pd.concat([ds1, ds2, ds3, ds4, ds5, ds6, ds7, ds8], ignore_index=True).sample(n=45000)
    # ds = ds.drop(['sTtl'], axis=1)
    # # Convert the ground truth Labels to numeric values
    # ds_1 = DataPrep().label_converter(ds)
    n_chunks = 3
    n_chunk_train = 2
    n_chunk_test = 1
    splitdata = DataSplit(n_chunks=n_chunks,
                     n_chunk_train=n_chunk_train,
                     n_chunk_test=n_chunk_test)
    train_df, test_df = splitdata.dataset_spit(ds, report_path)

    batch_sz = 3000
    init_batch_sz = 500
    # prepare the dictionary with configurations
    config_file = {'main_directory': dir_path,
                   'report_directory': report_path,
                   'batch_sz': batch_sz,
                   'init_batch_sz': init_batch_sz,
                   'n_chunks': n_chunks,
                   'n_chunk_train': n_chunk_train,
                   'n_chunk_test': n_chunk_test}
    # Iterate over different datasets
    for tr, ts in zip(train_df, test_df):
        itr_path = Methods().gen_folder(dir_path, f"Iteration_{tr}")
        print(f"*************************************Dataset_{tr}_{ts}**************************************", file=open(report_path, 'a'))
        # Scale dataset
        ds_train = DataPrep().dataset_cleaning(train_df[tr])
        train_file_path = Methods().gen_file(ds_train, f"{tr}.csv", itr_path, False)  # save training set
        ds_test = DataPrep().dataset_cleaning(test_df[ts])
        test_file_path = Methods().gen_file(ds_test, f"{ts}.csv", itr_path, False)  # save test set
        config_file[f"Iteration_path_{tr}"] = itr_path
        print(f"Train Dataset Dimension: {train_df[tr].shape}", file=open(report_path, 'a'))
        print(f"Test Dataset Dimension: {test_df[ts].shape}", file=open(report_path, 'a'))
        print(f"Initial training batch size: {init_batch_sz}", file=open(report_path, 'a'))
        print(f"Normal training batch size: {batch_sz}", file=open(report_path, 'a'))

    with open(r'config_directories.yaml', 'w') as file:
        config = yaml.dump(config_file, file)

    print("Initialization done. Please run individual script")

if __name__ == '__main__':
    main()

