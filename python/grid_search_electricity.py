import pandas as pd
import numpy as np
import random
random.seed(42)
from datetime import datetime, timedelta
from sklearn import preprocessing
from GBRT_for_TSF.utils import evaluate_with_xgboost
from mpmf.utils import get_top_1_motif_numba, get_top_1_motif_trend, get_top_k_motifs, compute_point_after_average

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
    start_date = datetime(2012, 1, 1, 00, 00, 00)  # define start date
    for i in range(0, len(TimeSeries)):
        record = []
        record.append(TimeSeries[i])  # adding the electricity value
        record.append(start_date.month)
        record.append(start_date.day)
        record.append(start_date.hour)
        record.append(start_date.weekday())
        record.append(start_date.timetuple().tm_yday)
        record.append(start_date.isocalendar()[1])
        start_date = start_date + timedelta(hours=1)
        Data.append(record)
    headers = [
        "electricity",
        "month",
        "day",
        "hour",
        "day_of_week",
        "day_of_year",
        "week_of_year",
    ]
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
        if include_motif_information == 1 or include_motif_information == 2:  # get_top_1_motif
            df_motif = get_top_1_motif_numba(
                TimeSeries,
                num_periods_output,
                l=no_points_after_motif,
                include_itself=include_itself,
                compute_trend=(include_motif_information == 2)
            )

            df_motif_points = df_motif[[c for c in df_motif.columns if (("idx" not in c) and ("dist" not in c))]]

            if do_normalization:
                df_motif_points = df_motif_points.sub(df_motif_points.mean(axis=1), axis=0).div(df_motif_points.std(axis=1), axis=0)
                df_motif_points = df_motif_points.replace([np.inf, -np.inf], np.nan)
            
            Normalized_Data_df = pd.concat([Normalized_Data_df, df_motif_points], axis=1)

            if include_similarity:
                df_motif_dist = df_motif[[c for c in df_motif.columns if ("dist" in c)]]
                df_motif_dist = pd.DataFrame(preprocessing.minmax_scale(df_motif_dist, feature_range=(-0.5, 0.5)), columns=df_motif_dist.columns, index=df_motif_dist.index)
                Normalized_Data_df = pd.concat([Normalized_Data_df, df_motif_dist], axis=1)

        # if include_motif_information == 2:  # get_top_k_motifs (Direct)
        #     df_motif = get_top_k_motifs(
        #         TimeSeries,
        #         num_periods_output,
        #         k=k_motifs,
        #         l=no_points_after_motif,
        #         include_itself=include_itself,
        #     )
        #     interested_features = [
        #         c for c in df_motif.columns if (("idx" not in c) and ("dist" not in c))
        #     ]
        #     df_motif = df_motif[interested_features]
        # if include_motif_information == 3:  # get_top_k_motifs (Unweighted Average)
        #     df_motif = get_top_k_motifs(
        #         TimeSeries,
        #         num_periods_output,
        #         k=k_motifs,
        #         l=no_points_after_motif,
        #         include_itself=include_itself,
        #     )
        #     interested_features = [c for c in df_motif.columns if ("idx" not in c)]
        #     df_motif = df_motif[interested_features]
        #     df_motif = compute_point_after_average(df_motif)
        # if include_motif_information == 4:  # get_top_k_motifs (Weighted Average)
        #     df_motif = get_top_k_motifs(
        #         TimeSeries,
        #         num_periods_output,
        #         k=k_motifs,
        #         l=no_points_after_motif,
        #         include_itself=include_itself,
        #     )
        #     interested_features = [c for c in df_motif.columns if ("idx" not in c)]
        #     df_motif = df_motif[interested_features]
        #     df_motif = compute_point_after_average(df_motif, method="weighted")

    #################################################################################################
    # Change 2
    split_index = 25968
    #################################################################################################
    Train = Normalized_Data_df.iloc[0:split_index, :]
    Train = Train.values
    Train = Train.astype("float32")
    # print('Traing length :',len(Train))
    Test = Normalized_Data_df.iloc[(split_index - num_periods_input) :, :]
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
    file_name = "electricity.npy"
    file_name_prefix = file_name.split('.')[0]
    data_path = r"../data/" + file_name
    data = np.load(data_path)
    data = data[0:70, :]
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
    "learning_rate": 0.2,
    "n_estimators": 150,
    "max_depth": 8,
    "min_child_weight": 1,
    "gamma": 0.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 1,
    "random_state": 42,
    "verbosity": 1, # 0=Silent, 1=Warning, 2=Info, 3=Debug
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
