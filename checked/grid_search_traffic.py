import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from sklearn import preprocessing
import sys
import os
import itertools

# Add current directory to path to ensure imports work
sys.path.append(os.getcwd())

try:
    from GBRT_for_TSF.utils import evaluate_with_xgboost
    from mpmf.utils import get_top_1_motif, get_top_k_motifs, compute_point_after_average
except ImportError:
    # If running from parent directory
    sys.path.append(os.path.join(os.getcwd(), 'python'))
    from GBRT_for_TSF.utils import evaluate_with_xgboost
    from mpmf.utils import get_top_1_motif, get_top_k_motifs, compute_point_after_average

random.seed(42)

def get_data():
    file_name = "traffic.npy"
    # Try to locate data
    possible_paths = [
        os.path.join("..", "data", file_name),
        os.path.join("data", file_name),
        os.path.join(os.path.dirname(__file__), "..", "data", file_name)
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
            
    if data_path is None:
        raise FileNotFoundError(f"Could not find {file_name} in expected locations.")
        
    print(f"Loading data from {data_path}")
    data = np.load(data_path)
    # Using the subset as in the notebook
    data = data[0:3, :]
    data = pd.DataFrame(data)
    return data

def preprocess_timeseries(TimeSeries, include_motif_information, k_motifs, no_points_after_motif, include_covariates, include_itself, num_periods_output, num_periods_input):
    Data = []
    start_date = datetime(2012, 1, 1, 00, 00, 00)
    for i in range(0, len(TimeSeries)):
        record = []
        record.append(TimeSeries[i])
        record.append(start_date.month)
        record.append(start_date.day)
        record.append(start_date.hour)
        record.append(start_date.weekday())
        record.append(start_date.timetuple().tm_yday)
        record.append(start_date.isocalendar()[1])
        start_date = start_date + timedelta(hours=1)
        Data.append(record)

    headers = [
        "traffic",
        "month",
        "day",
        "hour",
        "day_of_week",
        "day_of_year",
        "week_of_year",
    ]
    Data_df = pd.DataFrame(Data, columns=headers)
    sub = Data_df.iloc[:, 1:]
    New_sub = preprocessing.minmax_scale(sub, feature_range=(-0.5, 0.5))
    Normalized_Data_df = pd.DataFrame(
        np.column_stack([Data_df.iloc[:, 0], New_sub]), columns=headers
    )

    df_motif = None
    if include_motif_information:
        if include_motif_information == 1: # get_top_1_motif
            df_motif = get_top_1_motif(
                TimeSeries, num_periods_output, l=no_points_after_motif, include_itself=include_itself
            )
            interested_features = [
                c for c in df_motif.columns if (("idx" not in c) and ("dist" not in c))
            ]
            df_motif = df_motif[interested_features]
        if include_motif_information == 2: # get_top_k_motifs (Direct)
            df_motif = get_top_k_motifs(
                TimeSeries, num_periods_output, k=k_motifs, l=no_points_after_motif, include_itself=include_itself
            )
            interested_features = [
                c for c in df_motif.columns if (("idx" not in c) and ("dist" not in c))
            ]
            df_motif = df_motif[interested_features]
        if include_motif_information == 3: # get_top_k_motifs (Unweighted Average)
            df_motif = get_top_k_motifs(
                TimeSeries, num_periods_output, k=k_motifs, l=no_points_after_motif, include_itself=include_itself
            )
            interested_features = [c for c in df_motif.columns if ("idx" not in c)]
            df_motif = df_motif[interested_features]
            df_motif = compute_point_after_average(df_motif)
        if include_motif_information == 4: # get_top_k_motifs (Weighted Average)
            df_motif = get_top_k_motifs(
                TimeSeries, num_periods_output, k=k_motifs, l=no_points_after_motif, include_itself=include_itself
            )
            interested_features = [c for c in df_motif.columns if ("idx" not in c)]
            df_motif = df_motif[interested_features]
            df_motif = compute_point_after_average(df_motif, method="weighted")   
        
        New_df_motif = preprocessing.minmax_scale(df_motif, feature_range=(-0.5, 0.5))
        New_df_motif = pd.DataFrame(New_df_motif, columns=df_motif.columns, index=df_motif.index)
        Normalized_Data_df = pd.concat([Normalized_Data_df, New_df_motif], axis=1)

    Train = Normalized_Data_df.iloc[0:10392, :]
    Train = Train.values
    Train = Train.astype("float32")
    Test = Normalized_Data_df.iloc[10392 - num_periods_input :, :]
    Test = Test.values
    Test = Test.astype("float32")
    Number_Of_Features = Normalized_Data_df.shape[1]

    end = len(Train)
    start = 0
    next = 0
    x_batches = []
    y_batches = []
    while next + (num_periods_input) < end:
        next = start + num_periods_input
        x_batches.append(Train[start:next, :])
        y_batches.append(Train[next : next + num_periods_output, 0])
        start = start + 1
    y_batches = np.asarray(y_batches)
    y_batches = y_batches.reshape(-1, num_periods_output, 1)
    x_batches = np.asarray(x_batches)
    x_batches = x_batches.reshape(-1, num_periods_input, Number_Of_Features)

    end_test = len(Test)
    start_test = 0
    next_test = 0
    x_testbatches = []
    y_testbatches = []
    while next_test + (num_periods_input) < end_test:
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
        num_motif_cols = len(df_motif.columns.tolist())
        selected_cols = np.r_[0, len(headers) : len(headers) + num_motif_cols]
        x_batches = x_batches[:, :, selected_cols]
        x_testbatches = x_testbatches[:, :, selected_cols]
    else: # (not include_covariates) and (not include_motif_information)
        pass
    
    return x_batches, y_batches, x_testbatches, y_testbatches

def run_grid_search():
    # Parameters
    param_include_covariates = [True, False]
    param_include_motif_information = [0, 1, 2, 3, 4]
    param_k_motifs = [1, 2, 3]
    param_no_points_after_motif = [1, 3, 5]
    
    include_itself = True
    num_periods_output = 24
    num_periods_input = 24
    
    xgboost_parameters = {
        "learning_rate": 0.2,
        "n_estimators": 800,
        "max_depth": 8,
        "min_child_weight": 1,
        "gamma": 0.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 1,
        "random_state": 42,
        "verbosity": 0, # Silent
    }

    data = get_data()
    
    results = []

    # Create grid
    combinations = list(itertools.product(
        param_include_covariates,
        param_include_motif_information,
        param_k_motifs,
        param_no_points_after_motif
    ))
    
    total_combinations = len(combinations)
    print(f"Starting grid search with {total_combinations} combinations.")
    
    for idx, (cov, motif_info, k, points) in enumerate(combinations):
        print(f"[{idx+1}/{total_combinations}] Running: cov={cov}, motif={motif_info}, k={k}, points={points}")
        
        # Optimization: If motif_info == 0, k and points don't affect the result.
        # We can check if we already have a result for cov=X, motif=0
        # But for simplicity and to ensure full output structure, we run it.
        # Or we could run once and copy.
        # Let's just run it to be safe.
        
        x_batches_Full = []
        y_batches_Full = []
        X_Test_Full = []
        Y_Test_Full = []
        
        for i in range(0, len(data)):
            TimeSeries = data.iloc[i, :]
            x_b, y_b, x_t, y_t = preprocess_timeseries(
                TimeSeries, 
                motif_info, 
                k, 
                points, 
                cov, 
                include_itself, 
                num_periods_output, 
                num_periods_input
            )
            x_batches_Full.extend(x_b)
            y_batches_Full.extend(y_b)
            X_Test_Full.extend(x_t)
            Y_Test_Full.extend(y_t)
            
        rmse, wape, mae, mape = evaluate_with_xgboost(
            num_periods_output,
            x_batches_Full,
            y_batches_Full,
            X_Test_Full,
            Y_Test_Full,
            xgboost_parameters,
            (cov or (motif_info > 0)),
        )
        
        results.append({
            "include_covariates": cov,
            "include_motif_information": motif_info,
            "k_motifs": k,
            "no_points_after_motif": points,
            "RMSE": rmse,
            "WAPE": wape,
            "MAE": mae,
            "MAPE": mape
        })

    df_results = pd.DataFrame(results)
    output_file = "grid_search_results.csv"
    df_results.to_csv(output_file, index=False)
    print(f"Grid search completed. Results saved to {output_file}")

if __name__ == "__main__":
    run_grid_search()
