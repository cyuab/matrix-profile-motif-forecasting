# # Using GPU
# import os
# # Must be set before importing libraries that use the GPU!
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
# os.environ["CUDA_VISIBLE_DEVICES"] = "6" # Example: Only expose the RTX 5880 (Index 6)

import pandas as pd
import numpy as np
import random
random.seed(42)
from datetime import datetime, timedelta
from sklearn import preprocessing
from GBRT_for_TSF.utils import evaluate_with_xgboost
from mpmf.utils import get_top_1_motif_numba, get_top_k_motifs_numba

import itertools
import argparse  # Added for command line arguments

import time

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
    do_normalization=False,
    include_similarity=False
):
    Data = []
    # Change 1
    #################################################################################################
    start_date = datetime(1990, 1, 1, 00, 00, 00)  # define start date
    for i in range(0, len(TimeSeries)):
        record = []
        record.append(TimeSeries[i])  # adding the xchangerate value
        record.append(start_date.month)
        record.append(start_date.day)
        record.append(start_date.weekday())
        record.append(start_date.timetuple().tm_yday)
        record.append(start_date.isocalendar()[1])
        start_date = start_date + timedelta(days=1)
        Data.append(record)
    headers = ["pems", "month", "day", "day_of_week", "day_of_year", "week_of_year"]
    #################################################################################################
    Data_df = pd.DataFrame(Data, columns=headers)
    sub = Data_df.iloc[:, 1:]
    # Normalize features to be from -0.5 to 0.5 as mentioned in the paper
    New_sub = preprocessing.minmax_scale(sub, feature_range=(-0.5, 0.5))
    Normalized_Data_df = pd.DataFrame(
        np.column_stack([Data_df.iloc[:, 0], New_sub]), columns=headers
    )
    #################################################################################################
    if include_motif_information:
        # include_motif_information: 1:12; Odds: raw values; Evens: trend values
        if 1 <= include_motif_information <=12 :
            df_motif = get_top_1_motif_numba(
                    TimeSeries,
                    num_periods_input,
                    l=no_points_after_motif,
                    include_itself=include_itself,
                    compute_trend=(include_motif_information == 2 or include_motif_information == 4 or include_motif_information == 6 or include_motif_information == 8 or include_motif_information == 10 or include_motif_information == 12)
                )
        else:
            df_motif = get_top_k_motifs_numba(TimeSeries, num_periods_input, k_motifs, l=no_points_after_motif, include_itself=include_itself)
        df_motif_points_after = df_motif[[c for c in df_motif.columns if ("point_after" in c)]]
        if do_normalization:
            df_motif_points_after = df_motif_points_after.sub(df_motif_points_after.mean(axis=1), axis=0).div(df_motif_points_after.std(axis=1, ddof=0), axis=0) # Use Population Standard Deviation (ddof=0)
            df_motif_points_after = df_motif_points_after.replace([np.inf, -np.inf], np.nan)
        Normalized_Data_df = pd.concat([Normalized_Data_df, df_motif_points_after], axis=1)
        # Add motif points before
        if include_motif_information in [3, 4, 5, 6, 9, 10, 11, 12, 15 ,17, 21, 23]: # No 7, 8
            df_motif_points_before = df_motif[[c for c in df_motif.columns if ("point_before" in c)]]
            if do_normalization:
                df_motif_points_before = df_motif_points_before.sub(df_motif_points_before.mean(axis=1), axis=0).div(df_motif_points_before.std(axis=1, ddof=0), axis=0) # Use Population Standard Deviation (ddof=0)
                df_motif_points_before = df_motif_points_before.replace([np.inf, -np.inf], np.nan)
            if include_motif_information in [3, 4, 9, 10, 15, 21]:
                # print("df_motif_points_before shape before slicing:", df_motif_points_before.shape)
                df_motif_points_before = df_motif_points_before.iloc[:, -1:]  # only last point
                # print("df_motif_points_before shape after slicing:", df_motif_points_before.shape)
            else:
                pass  # all y points
            Normalized_Data_df = pd.concat([Normalized_Data_df, df_motif_points_before], axis=1)
        if (7 <= include_motif_information <= 12) or (include_motif_information in [19, 21, 23]):
            current_k = 1
            while f"top_{current_k}_motif_idx" in df_motif.columns:
                col_name = f"top_{current_k}_motif_idx"
                current_idx_col = df_motif[[col_name]]
                
                last_point_idx = current_idx_col + num_periods_input - 1
                
                # Retrieve time features, handling NaNs in last_point_idx
                idx_values = last_point_idx.values.flatten()
                valid_mask = np.isfinite(idx_values)
                
                # Create container filled with NaNs
                motif_time_features = np.full((len(idx_values), New_sub.shape[1]), np.nan)
                
                if np.any(valid_mask):
                    valid_indices = idx_values[valid_mask].astype(int)
                    motif_time_features[valid_mask] = New_sub[valid_indices]

                suffix = "" if current_k == 1 else f"_{current_k}"
                
                motif_feat_df = pd.DataFrame(motif_time_features, columns=[
                    f"month_motif{suffix}", f"day_motif{suffix}",
                    f"day_of_week_motif{suffix}", f"day_of_year_motif{suffix}", f"week_of_year_motif{suffix}"
                ])
                Normalized_Data_df = pd.concat([Normalized_Data_df, motif_feat_df], axis=1)
                
                current_k += 1


        if include_similarity:
            df_motif_dist = df_motif[[c for c in df_motif.columns if ("dist" in c)]]
            # print("Shape of df_motif_dist before normalization:", df_motif_dist.shape)
            df_motif_dist = pd.DataFrame(preprocessing.minmax_scale(df_motif_dist, feature_range=(-0.5, 0.5)), columns=df_motif_dist.columns, index=df_motif_dist.index)
            # print("Shape of df_motif_dist after normalization:", df_motif_dist.shape)
            Normalized_Data_df = pd.concat([Normalized_Data_df, df_motif_dist], axis=1)

    #################################################################################################
    # Change 2
    # cut training and testing
    train_split = np.floor(len(Normalized_Data_df) * 0.8)  # 60 % training
    train_split = int(
        train_split - (train_split % (num_periods_output + num_periods_input))
    )
    Train = Normalized_Data_df.iloc[0:train_split, :]
    Train = Train.values
    Train = Train.astype("float32")
    # print('Traing length :',len(Train))
    total = len(Normalized_Data_df)
    test_split = np.floor(len(Normalized_Data_df) * 0.2)  # 20 % testing
    test_split = int(
        test_split - (test_split % (num_periods_output + num_periods_input))
    )
    Test = Normalized_Data_df.iloc[(total - test_split - num_periods_input) :, :]
    Test = Test.values
    Test = Test.astype("float32")
    # print('Traing length :',len(Test))
    # Number_Of_Features = 8
    Number_Of_Features = Normalized_Data_df.shape[1]
    #################################################################################################
    ############################################ Windowing ##################################
    end = len(Train)
    start = 0
    next = 0
    x_batches = []
    y_batches = []
    limit = max(num_periods_input, num_periods_output)
    while next + limit < end:
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
    while next_test + limit < end_test:
        next_test = start_test + num_periods_input
        x_testbatches.append(Test[start_test:next_test, :])
        y_testbatches.append(Test[next_test : next_test + num_periods_output, 0])
        start_test = start_test + num_periods_input
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
            0, len(headers) : Normalized_Data_df.shape[1]
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
    param_no_points_after_motif,
    param_do_normalization,
    param_include_similarity,
    param_no_time_series,
):
    # --- Warmup Run ---
    print("Performing Numba warmup (ignoring time)...")
    # This compiles the function so the next run is fast
    _ = get_top_1_motif_numba(np.random.rand(100), m=10)
    print("Warmup complete.\n")
    # Change 3
    #################################################################################################
    file_name = "exchange_rate.txt"
    file_name_prefix = file_name.split('.')[0]
    data_path = r"../data/" + file_name
    data = pd.read_csv(data_path, sep=",", header=None)
    data = pd.DataFrame(data)
    data = data.T
    #################################################################################################
    data = pd.DataFrame(data)
    if param_no_time_series[0] != -1:
        data = data[: param_no_time_series[0]]
    print("Data shape:", data.shape)

    # Change 4
    #################################################################################################
    num_periods_input = 24  # input
    num_periods_output = 24  # to predict
    #################################################################################################

    # Change 5
    #################################################################################################
    xgboost_parameters = {
        "learning_rate": 0.07,
        "n_estimators": 80,
        "max_depth": 3,
        "min_child_weight": 1,
        "gamma": 0.0,
        "subsample": 0.97,
        "colsample_bytree": 0.97,
        "scale_pos_weight": 1,
        "random_state": 42,
        "verbosity": 1, # 0=Silent, 1=Warning, 2=Info, 3=Debug
        # # For GPU acceleration
        # "device": "cuda:0",  # Uses the first GPU
        # "tree_method": "hist" # Required for modern GPU acceleration
    }
    #################################################################################################
    results = []

    # Create grid
    combinations = list(
        itertools.product(
            param_include_covariates,
            param_include_itself,
            param_include_motif_information,
            param_k_motifs,
            param_no_points_after_motif,
            param_do_normalization,
            param_include_similarity,
        )
    )

    total_combinations = len(combinations)
    print(f"Starting grid search with {total_combinations} combinations.")

    for idx, (
        include_covariates,
        include_itself,
        include_motif_information,
        k_motifs,
        no_points_after_motif,
        do_normalization,
        include_similarity,
    ) in enumerate(combinations):
        start_time = time.time()
        print(
            f"[{idx+1}/{total_combinations}] Running: include_covariates={include_covariates}, include_itself={include_itself}, include_motif_information={include_motif_information}, k_motifs={k_motifs}, no_points_after_motif={no_points_after_motif}, do_normalization={do_normalization}, include_similarity={include_similarity}"
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
                include_covariates,
                include_motif_information,
                k_motifs,
                no_points_after_motif,
                do_normalization,
                include_similarity
            )
            for element1 in x_batches:
                x_batches_Full.append(element1)

            for element2 in y_batches:
                y_batches_Full.append(element2)

            for element5 in X_Test:
                X_Test_Full.append(element5)

            for element6 in Y_Test:
                Y_Test_Full.append(element6)
        end_time_matrix_profile = time.time()
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
        end_time = time.time()
        # print(f"Total execution time: {end_time - start_time:.2f} seconds")
        results.append({
            "include_covariates": include_covariates,
            "include_itself": include_itself,
            "include_motif_information": include_motif_information,
            "k_motifs": k_motifs,
            "no_points_after_motif": no_points_after_motif,
            "do_normalization": do_normalization,
            "include_similarity": include_similarity,
            "RMSE": rmse,
            "WAPE": wape,
            "MAE": mae,
            "MAPE": mape,
            "Time_Processing": end_time_matrix_profile - start_time,
            "Time_Overall": end_time - start_time
        })

    # ======= Save results =======
    df_results = pd.DataFrame(results)
    
    output_file = (
        "../results/grid_search_"+file_name_prefix+"_results_"
        + str(param_include_covariates)
        + "_"
        + str(include_itself)
        + "_"
        + str(param_include_motif_information)
        + "_"
        + str(param_k_motifs)
        + "_"
        + str(param_no_points_after_motif)
        + "_"
        + str(param_do_normalization)
        + "_"
        + str(param_include_similarity)
        + ".csv"
    )
    df_results.to_csv(output_file, index=False)
    # ======= End of Save results =======
    print(f"Grid search completed. Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid Search")

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
    parser.add_argument("--do_normalization", nargs='+', type=str2bool, default=[False], 
                        help="List of booleans for do_normalization")
    parser.add_argument("--include_similarity", nargs='+', type=str2bool, default=[False], 
                        help="List of booleans for include_similarity")
    parser.add_argument("--no_time_series", nargs='+', type=int, default=[-1], 
                        help="List of integers for number of time series")

    args = parser.parse_args()
    print("Starting grid search...")
    # run_grid_search()
    run_grid_search(
        param_include_covariates=args.include_covariates,
        param_include_itself=args.include_itself,
        param_include_motif_information=args.include_motif_information,
        param_k_motifs=args.k_motifs,
        param_no_points_after_motif=args.no_points_after_motif,
        param_do_normalization=args.do_normalization,
        param_include_similarity=args.include_similarity,
        param_no_time_series=args.no_time_series
    )
