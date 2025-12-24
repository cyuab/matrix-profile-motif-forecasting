import numpy as np
import random
random.seed(42)

import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor


def evaluate_with_xgboost(
    num_periods_output,
    x_batches_Full,
    y_batches_Full,
    X_Test_Full,
    Y_Test_Full,
    xgboost_parameters,
    include_covariates =True,
):
    # ---------------------shuffle windows  X and target Y together-------------------------------------
    combined = list(zip(x_batches_Full, y_batches_Full))
    random.shuffle(combined)
    shuffled_batch_features, shuffled_batch_y = zip(*combined)

    # xgboost part
    All_Training_Instances = []

    # =============== flatten each training window into Instance =================================
    for i in range(0, len(shuffled_batch_features)):
        hold = []
        # temp = []
        for j in range(0, len(shuffled_batch_features[i])):
            # **************** to run without features -->comment if else condition (just keep else statement) **************************
            if include_covariates:
                if j == (len(shuffled_batch_features[i]) - 1):
                    hold = np.concatenate(
                        (hold, shuffled_batch_features[i][j][:]), axis=None
                    )
                else:
                    hold = np.concatenate(
                        (hold, shuffled_batch_features[i][j][0]), axis=None
                    )
            else:
                hold = np.concatenate(
                        (hold, shuffled_batch_features[i][j][0]), axis=None
                    )

        All_Training_Instances.append(hold)

    # =============== change each window into Instance =================================
    All_Testing_Instances = []
    for i in range(0, len(X_Test_Full)):
        hold = []
        # temp = []
        for j in range(0, len(X_Test_Full[i])):
            # **************** to run without features -->comment if else condition (just keep else statement) **************************
            if include_covariates:
                if j == (len(X_Test_Full[i]) - 1):
                    hold = np.concatenate((hold, X_Test_Full[i][j][:]), axis=None)
                else:
                    hold = np.concatenate((hold, X_Test_Full[i][j][0]), axis=None)
            else:
                hold = np.concatenate(
                        (hold, shuffled_batch_features[i][j][0]), axis=None
                    )

        All_Testing_Instances.append(hold)

    # =========================== final shape check =========================
    All_Testing_Instances = np.reshape(
        All_Testing_Instances,
        (len(All_Testing_Instances), len(All_Testing_Instances[0])),
    )
    Y_Test_Full = np.reshape(Y_Test_Full, (len(Y_Test_Full), num_periods_output))

    All_Training_Instances = np.reshape(
        All_Training_Instances,
        (len(All_Training_Instances), len(All_Training_Instances[0])),
    )
    shuffled_batch_y = np.reshape(
        shuffled_batch_y, (len(shuffled_batch_y), num_periods_output)
    )

    # =========================== CALLING XGBOOST ===========================
    model = xgb.XGBRegressor(**xgboost_parameters)

    multioutput = MultiOutputRegressor(model).fit(All_Training_Instances, shuffled_batch_y)

    # ============================== PREDICTION ===============================
    prediction = multioutput.predict(All_Testing_Instances)
    MSE = np.mean((prediction - Y_Test_Full) ** 2)
    WAPE = np.sum(np.abs(prediction - Y_Test_Full)) / np.sum(np.abs(Y_Test_Full))
    MAE = np.mean(np.abs((prediction - Y_Test_Full)))
    MAPE = np.mean((np.abs(prediction - Y_Test_Full) / np.abs(Y_Test_Full)))
    print("RMSE: ", MSE**0.5)
    print("WAPE: ", WAPE)
    print("MAE: ", MAE)
    print('MAPE: ',MAPE)