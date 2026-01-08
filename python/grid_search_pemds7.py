"""
Grid Search for PEMDS7 Dataset

Usage Example:
    python grid_search_pemds7.py \
      --include_covariates True False \
      --include_itself True \
      --include_motif_information 0 \
      --k_motifs 1 \
      --no_points_after_motif 1
"""

import pandas as pd
import numpy as np
import random

random.seed(42)
from datetime import datetime, timedelta
from sklearn import preprocessing

from GBRT_for_TSF.utils import evaluate_with_xgboost
from mpmf.utils import get_top_1_motif, get_top_k_motifs, compute_point_after_average

import sys
import os
import itertools
import argparse  # Added for command line arguments

# Helper function to convert string inputs to booleans
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def New_preprocessing(
    TimeSeries,
    num_periods_input,
    num_periods_output,
    include_itself,
    include_covariates,
    include_motif_information,
    k_motifs,
    no_points_after_motif,
):
    # print(len(TimeSeries))
    Data = []
    start_date = datetime(2012, 5, 1, 00, 00, 00)  # define start date
    for i in range(0, len(TimeSeries)):
        record = []
        record.append(TimeSeries[i])  # adding the pemds7 value
        record.append(start_date.month)
        record.append(start_date.day)
        record.append(start_date.hour)
        record.append(start_date.minute)
        record.append(start_date.weekday())
        record.append(start_date.timetuple().tm_yday)
        record.append(start_date.isocalendar()[1])
        start_date = start_date + timedelta(minutes=5)
        Data.append(record)
    ########## change list of lists to df ################
    headers = [
        "pems",
        "month",
        "day",
        "hour",
        "minute",
        "day_of_week",
        "day_of_year",
        "week_of_year",
    ]
    Data_df = pd.DataFrame(Data, columns=headers)
    sub = Data_df.iloc[:, 1:]
    # Normalize features to be from -0.5 to 0.5 as mentioned in the paper
    New_sub = preprocessing.minmax_scale(sub, feature_range=(-0.5, 0.5))
    Normalized_Data_df = pd.DataFrame(
        np.column_stack([Data_df.iloc[:, 0], New_sub]), columns=headers
    )
    if include_motif_information:
        if include_motif_information == 1:  # get_top_1_motif
            df_motif = get_top_1_motif(
                TimeSeries,
                num_periods_output,
                l=no_points_after_motif,
                include_itself=include_itself,
            )
            interested_features = [
                c for c in df_motif.columns if (("idx" not in c) and ("dist" not in c))
            ]
            df_motif = df_motif[interested_features]
        if include_motif_information == 2:  # get_top_k_motifs (Direct)
            df_motif = get_top_k_motifs(
                TimeSeries,
                num_periods_output,
                k=k_motifs,
                l=no_points_after_motif,
                include_itself=include_itself,
            )
            interested_features = [
                c for c in df_motif.columns if (("idx" not in c) and ("dist" not in c))
            ]
            df_motif = df_motif[interested_features]
        if include_motif_information == 3:  # get_top_k_motifs (Unweighted Average)
            df_motif = get_top_k_motifs(
                TimeSeries,
                num_periods_output,
                k=k_motifs,
                l=no_points_after_motif,
                include_itself=include_itself,
            )
            interested_features = [c for c in df_motif.columns if ("idx" not in c)]
            df_motif = df_motif[interested_features]
            df_motif = compute_point_after_average(df_motif)
        if include_motif_information == 4:  # get_top_k_motifs (Weighted Average)
            df_motif = get_top_k_motifs(
                TimeSeries,
                num_periods_output,
                k=k_motifs,
                l=no_points_after_motif,
                include_itself=include_itself,
            )
            interested_features = [c for c in df_motif.columns if ("idx" not in c)]
            df_motif = df_motif[interested_features]
            df_motif = compute_point_after_average(df_motif, method="weighted")
        # Normailize motif features to be from -0.5 to 0.5
        New_df_motif = preprocessing.minmax_scale(df_motif, feature_range=(-0.5, 0.5))
        # Convert the numpy array back to a DataFrame using the original columns and index
        New_df_motif = pd.DataFrame(
            New_df_motif, columns=df_motif.columns, index=df_motif.index
        )
        Normalized_Data_df = pd.concat([Normalized_Data_df, New_df_motif], axis=1)
    # print(Normalized_Data_df)
    #################################################################################################
    # cut training and testing training is 11232
    Train = Normalized_Data_df.iloc[0:11232, :]
    Train = Train.values
    Train = Train.astype("float32")
    # print('Traing length :',len(Train))
    Test = Normalized_Data_df.iloc[(11232 - num_periods_input) :, :]
    Test = Test.values
    Test = Test.astype("float32")
    # print('Traing length :',len(Test))
    # Number_Of_Features = 8
    Number_Of_Features = Normalized_Data_df.shape[1]
    ############################################ Windowing ##################################
    end = len(Train)
    start = 0
    next = 0
    x_batches = []
    y_batches = []
    while next + (num_periods_input + num_periods_output) < end:
        next = start + num_periods_input
        x_batches.append(Train[start:next, :])
        y_batches.append(Train[next : next + num_periods_output, 0])
        start = start + 1
    y_batches = np.asarray(y_batches)
    y_batches = y_batches.reshape(-1, num_periods_output, 1)
    x_batches = np.asarray(x_batches)
    x_batches = x_batches.reshape(-1, num_periods_input, Number_Of_Features)
    ############################################ Windowing ##################################
    end_test = len(Test)
    start_test = 0
    next_test = 0
    x_testbatches = []
    y_testbatches = []
    while next_test + (num_periods_input + num_periods_output) < end_test:
        next_test = start_test + num_periods_input
        x_testbatches.append(Test[start_test:next_test, :])
        y_testbatches.append(Test[next_test : next_test + num_periods_output, 0])
        start_test = start_test + 1
    y_testbatches = np.asarray(y_testbatches)
    y_testbatches = y_testbatches.reshape(-1, num_periods_output, 1)
    x_testbatches = np.asarray(x_testbatches)
    x_testbatches = x_testbatches.reshape(-1, num_periods_input, Number_Of_Features)
    if include_covariates and include_motif_information:
        pass
    elif include_covariates and (not include_motif_information):
        pass
    elif (not include_covariates) and include_motif_information:
        selected_cols = np.r_[
            0, len(headers) : len(headers) + len(df_motif.columns.tolist())
        ]
        x_batches = x_batches[:, :, selected_cols]
        x_testbatches = x_testbatches[:, :, selected_cols]
    else: # (not include_covariates) and (not include_motif_information)
        pass
    return x_batches, y_batches, x_testbatches, y_testbatches

def run_grid_search(
    param_include_covariates,
    param_include_itself,
    param_include_motif_information,
    param_k_motifs,
    param_no_points_after_motif
):
    file_name = "pems.npy"
    data_path = r"../data/" + file_name
    data = np.load(data_path)
    data = pd.DataFrame(data)

    num_periods_input = 9  # input
    num_periods_output = 9  # to predict

    # # ================== Parameters that are going to be changed==================
    # param_include_covariates = [
    #     True, False
    # ]  # True/False : whether to include features or not
    # param_include_itself = [True]
    # param_include_motif_information = [
    #     1
    # ]  # 0: no motif info; 1: Top-1 Motif; 2: Top-K Motifs (Direct); 3: Top-K Motifs (Average); 4: Top-K Motifs (Weighted Average)
    # param_k_motifs = [1]  # 1, 3, 5
    # param_no_points_after_motif = [1]  # number of points to consider: 1, ceil(m/2), m
    # # ================================================

    xgboost_parameters = {
        "learning_rate": 0.045,
        "n_estimators": 150,
        "max_depth": 8,
        "min_child_weight": 1,
        "gamma": 0.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 1,
        "random_state": 42,
        "verbosity": 1,  # 0=Silent, 1=Warning, 2=Info, 3=Debug
    }

    results = []

    # Create grid
    combinations = list(
        itertools.product(
            param_include_covariates,
            param_include_itself,
            param_include_motif_information,
            param_k_motifs,
            param_no_points_after_motif,
        )
    )

    total_combinations = len(combinations)
    print(f"Starting grid search with {total_combinations} combinations.")

    for idx, (
        include_covariates,
        include_itself,
        include_covariates,
        include_motif_information,
        k_motifs,
        no_points_after_motif,
    ) in enumerate(combinations):
        print(
            f"[{idx+1}/{total_combinations}] Running: include_covariates={include_covariates}, include_itself={include_itself}, include_motif_information={include_motif_information}, k_motifs={k_motifs}, no_points_after_motif={no_points_after_motif}"
        )
        # Reset seeds to ensure each run is independent (fresh start)
        random.seed(42)
        np.random.seed(42)

        # =================== Processing Time Series ===================
        x_batches_Full = []
        y_batches_Full = []
        X_Test_Full = []
        Y_Test_Full = []
        for i in range(0, len(data)):
            # print("Processing Time Series:", i)
            x_batches = []
            y_batches = []
            X_Test = []
            Y_Test = []
            TimeSeries = data.iloc[i, :]
            x_batches, y_batches, X_Test, Y_Test = New_preprocessing(
                TimeSeries,
                num_periods_input,
                num_periods_output,
                include_itself,
                include_motif_information,
                k_motifs,
                no_points_after_motif,
            )
            for element1 in x_batches:
                x_batches_Full.append(element1)

            for element2 in y_batches:
                y_batches_Full.append(element2)

            for element5 in X_Test:
                X_Test_Full.append(element5)

            for element6 in Y_Test:
                Y_Test_Full.append(element6)
        ## =================== End of Processing Time Series ===================
        rmse, wape, mae, mape = evaluate_with_xgboost(
            num_periods_output,
            x_batches_Full,
            y_batches_Full,
            X_Test_Full,
            Y_Test_Full,
            xgboost_parameters,
            (include_covariates or (include_motif_information > 0)),
        )
        results.append({
            "include_covariates": include_covariates,
            "include_itself": include_itself,
            "include_motif_information": include_motif_information,
            "k_motifs": k_motifs,
            "no_points_after_motif": no_points_after_motif,
            "RMSE": rmse,
            "WAPE": wape,
            "MAE": mae,
            "MAPE": mape
        })

    # ======= Save results =======
    df_results = pd.DataFrame(results)
    output_file = (
        "../results/grid_search_pemds7_results_"
        + str(param_include_covariates)
        + "_"
        + str(include_itself)
        + "_"
        + str(param_include_motif_information)
        + "_"
        + str(param_k_motifs)
        + "_"
        + str(param_no_points_after_motif)
        + ".csv"
    )
    df_results.to_csv(output_file, index=False)
    # ======= End of Save results =======
    print(f"Grid search completed. Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid Search for PEMDS7")

    # Accepting lists of arguments using nargs='+'
    # type=str2bool ensures 'True'/'False' strings are converted to booleans
    parser.add_argument("--include_covariates", nargs='+', type=str2bool, default=[True, False], 
                        help="List of booleans for include_covariates")
    parser.add_argument("--include_itself", nargs='+', type=str2bool, default=[True], 
                        help="List of booleans for include_itself")
    parser.add_argument("--include_motif_information", nargs='+', type=int, default=[0], 
                        help="List of integers for motif information types")
    parser.add_argument("--k_motifs", nargs='+', type=int, default=[1], 
                        help="List of integers for k motifs")
    parser.add_argument("--no_points_after_motif", nargs='+', type=int, default=[1], 
                        help="List of integers for points after motif")

    args = parser.parse_args()
    print("Starting grid search for PEMDS7 dataset...")
    # run_grid_search()
    run_grid_search(
        param_include_covariates=args.include_covariates,
        param_include_itself=args.include_itself,
        param_include_motif_information=args.include_motif_information,
        param_k_motifs=args.k_motifs,
        param_no_points_after_motif=args.no_points_after_motif
    )
