import numpy as np
import pandas as pd
import stumpy
from stumpy import config, core


def get_top_1_motif(T, m, l=1, include_itself=False):
    """Get the motif information for the time series T.

    Parameters
    ----------
    T : np.ndarray
        The time series data.
    m : int
        The length of the motifs to find.
    l : int
        The number of points after the motif to consider.
    include_itself : bool
        Whether to include the motif itself in the results.

    Returns
    -------
    tuple
        A tuple containing:
        - top_1_motif_idx: Indices of the top-1 motifs.
        - top_1_motif_idx_delta: Distance from the query to the top-1 motif.
        - top_1_motif_dist: Distances of the top-1 motifs.
        - top_1_motif_points_after: Points after the motifs.
    """
    # print("get_top_1_motif")
    mp = stumpy.stump(T, m=m)

    top_1_motif_dist = np.full(len(T), np.nan)
    top_1_motif_idx = np.full(len(T), np.nan)
    top_1_motif_idx_delta = np.full(len(T), np.nan)
    top_1_motif_points_after = np.full((len(T), l), np.nan)

    if include_itself:
        first_i_idx = m - 1
    else:
        first_i_idx = m

    for i in range(first_i_idx, len(T)):
        if include_itself:
            q_idx = i - (m - 1)
        else:
            q_idx = i - m

        j = mp.left_I_[q_idx]
        if j >= 0:  # We have found a top-1 motif of q
            top_1_motif_idx[i] = j
            # Single point
            # if j + m < len(T):
            #     top_1_motif_point_after[i] = T[j + m]
            # l points after the motif
            for ll in range(l):
                tgt_idx = j + m + ll
                if tgt_idx < len(T):
                    top_1_motif_points_after[i, ll] = T[tgt_idx]
            top_1_motif_idx_delta[i] = (
                q_idx - top_1_motif_idx[i]
            )  # how far the top-1 motif is from the query.
            query = T[
                q_idx : q_idx + m
            ].to_numpy()  # The distance calculation requires they are in NumPy arrays.
            top_1_motif = T[j : j + m].to_numpy()
            top_1_motif_dist[i] = np.linalg.norm(
                stumpy.core.z_norm(query) - stumpy.core.z_norm(top_1_motif)
            )
            top_1_motif_point_after_cols = {
                f"top_1_motif_point_after_{ll+1}": top_1_motif_points_after[:, ll]
                for ll in range(l)
            }
            df_motif = pd.DataFrame(
                {
                    # "top_1_motif_dist": top_1_motif_dist,
                    # "top_1_motif_idx": top_1_motif_idx,
                    # "top_1_motif_idx_delta": top_1_motif_idx_delta,
                    **top_1_motif_point_after_cols,
                }
            )
    return df_motif


def get_top_k_motifs(T, m, k=1, l=1, include_itself=False):
    # https://github.com/stumpy-dev/stumpy/discussions/1093#discussioncomment-14063985

    print("get_top_k_motifs")
    top_k_motif_dist = np.full((len(T), k), np.nan)
    top_k_motif_idx = np.full((len(T), k), np.nan)
    top_k_motif_idx_delta = np.full((len(T), k), np.nan)
    top_k_motif_points_after = np.full((len(T), k, l), np.nan)

    if include_itself:
        first_i_idx = m - 1
    else:
        first_i_idx = m
    for i in range(first_i_idx, len(T)):
        if include_itself:
            stop_idx = i + 1
        else:
            stop_idx = i
        start_idx = stop_idx - m
        T_prime = T[start_idx:stop_idx]
        # print(f"i={i}, start_idx={start_idx}, stop_idx={stop_idx}, T_prime={T_prime}")
        T_copy = T.copy()  # Make a copy of T to keep the original intact
        # Apply exclusion zone
        # https://github.com/stumpy-dev/stumpy/blob/534488d0b84f2bc20d529e6c46daf62c497f5f2b/stumpy/core.py#L2078
        excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
        # Clear all values to the right of the start index
        T_copy[start_idx - excl_zone - 1 + m :] = np.nan
        matches = stumpy.match(T_prime, T_copy, max_matches=k)
        # print(f"i={i}, matches={matches}")
        for kk in range(len(matches)):
            top_k_motif_dist[i, kk] = matches[kk, 0]
            top_k_motif_idx[i, kk] = matches[kk, 1]
            top_k_motif_idx_delta[i, kk] = start_idx - top_k_motif_idx[i, kk]
            top_k_motif_points_after[i, kk, :] = T[
                matches[kk, 1] + m : matches[kk, 1] + m + l
            ]

    top_k_motif_dist_cols = {
        f"top_{kk+1}_motif_dist": top_k_motif_dist[:, kk] for kk in range(k)
    }
    top_k_motif_idx_cols = {
        f"top_{kk+1}_motif_idx": top_k_motif_idx[:, kk] for kk in range(k)
    }
    top_k_motif_idx_delta_cols = {
        f"top_{kk+1}_motif_idx_delta": top_k_motif_idx_delta[:, kk] for kk in range(k)
    }
    top_k_motif_point_after_cols = {}
    for kk in range(k):
        top_k_motif_point_after_cols.update(
            {
                f"top_{kk+1}_motif_point_after_{ll+1}": top_k_motif_points_after[
                    :, kk, ll
                ]
                for ll in range(l)
            }
        )
    df_motif = pd.DataFrame(
        {
            # **top_k_motif_dist_cols,
            # **top_k_motif_idx_cols,
            # **top_k_motif_idx_delta_cols,
            **top_k_motif_point_after_cols,
        }
    )
    return df_motif


def mean_absolute_percentage_error(y_true, y_pred):
    a = y_true - y_pred
    b = y_true
    c = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    return np.mean(np.abs(c)) * 100


def print_prediction_results(prediction, Y_Test_Full):
    # print('prediction ',prediction.shape)
    # print('test ',Y_Test_Full.shape)
    MSE = np.mean((prediction - Y_Test_Full) ** 2)
    MAE = np.mean(np.abs((prediction - Y_Test_Full)))
    MAPE = mean_absolute_percentage_error(Y_Test_Full, prediction)
    WAPE = np.sum(np.abs(prediction - Y_Test_Full)) / np.sum(np.abs(Y_Test_Full))
    # print('With Features for {} weeks'.format(No_Of_weeks))
    # print('With Features for {} weeks'.format(No_Of_weeks))

    print("RMSE: ", MSE**0.5)
    print("WAPE: ", WAPE)
    print("MAE: ", MAE)
    # print('MAPE: ',MAPE)


def print_hello_world():
    print("Hello, world!")
