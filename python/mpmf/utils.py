import numpy as np
import pandas as pd
import stumpy
from stumpy import config, core
import re
import warnings
# import sys, os
from numba import njit
# from numba import set_num_threads, get_num_threads
# set_num_threads(8)

@njit(cache=True)
def z_norm(a):
    mu = np.mean(a)
    sigma = np.std(a)
    if sigma == 0:
        return np.zeros_like(a)
    return (a - mu) / sigma


@njit(cache=True)
def _extract_motif_data_numba(T, mp_left_I, m, l, start_idx_loop, q_offset, compute_trend):

    n = len(T)

    top_1_idx = np.full(n, np.nan, dtype=np.float64)
    top_1_dist = np.full(n, np.nan, dtype=np.float64)
    top_1_delta = np.full(n, np.nan, dtype=np.float64)
    top_1_after = np.full((n, l), np.nan, dtype=np.float64)
    top_1_before = np.full((n, m), np.nan, dtype=np.float64)

    for i in range(start_idx_loop, n):
        q_idx = i - q_offset

        # Check bounds for query index
        if q_idx >= 0 and q_idx < len(mp_left_I):

            idx_neighbor = mp_left_I[q_idx]

            # Stumpy uses -1 to indicate no neighbor found
            if idx_neighbor >= 0:
                top_1_idx[i] = int(idx_neighbor)

                query = T[q_idx : q_idx + m]
                neighbor = T[idx_neighbor : idx_neighbor + m]
                top_1_dist[i] = np.linalg.norm(z_norm(query) - z_norm(neighbor))
                top_1_delta[i] = int(q_idx - idx_neighbor)

                # Calculate where the nearest neighbor ends
                neighbor_end = idx_neighbor + m

                # Extract 'l' points after the nearest neighbor
                for ll in range(l):
                    tgt_idx = neighbor_end + ll # tgt_idx: target index
                    if tgt_idx < n:
                        if compute_trend:
                            top_1_after[i, ll] = T[tgt_idx] - T[tgt_idx - 1]
                        else:
                            top_1_after[i, ll] = T[tgt_idx]
                # Extract `m`points of the nearest neighbor
                for mm in range(m):
                    tgt_idx = idx_neighbor + mm
                    if tgt_idx < n:
                        top_1_before[i, mm] = T[tgt_idx]
                

    return top_1_idx, top_1_dist, top_1_delta, top_1_after, top_1_before

def get_top_1_motif_numba(T, m, l=1, compute_trend = False, include_itself=False):

    # Convert Pandas Series to Numpy if needed
    if isinstance(T, pd.Series):
        T = T.values
    # Ensure Float64 for Numba compatibility
    T = T.astype(np.float64)

    mp = stumpy.stump(T, m=m, ignore_trivial=True)

    if include_itself:
        start_idx_loop = m - 1
        q_offset = m - 1
    else:
        start_idx_loop = m
        q_offset = m

    top_1_idx, top_1_dist, top_1_delta, top_1_after, top_1_before = _extract_motif_data_numba(
        T, mp.left_I_, int(m), int(l), int(start_idx_loop), int(q_offset), compute_trend
    )

    # --- Format DataFrame ---
    result_dict = {
        "top_1_motif_dist": top_1_dist,
        "top_1_motif_idx": top_1_idx,
        "top_1_motif_idx_delta": top_1_delta,
    }

    for ll in range(l):
        result_dict[f"top_1_motif_point_after_{ll+1}"] = top_1_after[:, ll]
    for mm in range(m):
        result_dict[f"top_1_motif_point_before_{mm+1}"] = top_1_before[:, mm]

    return pd.DataFrame(result_dict)

def get_top_1_motif_slow(T, m, l=1, include_itself=False):
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

    mp = stumpy.stump(T, m=m, ignore_trivial=True)
    # mp = stumpy.gpu_stump(T, m=m, ignore_trivial=True)
    
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
            ]
            # query = T[
            #     q_idx : q_idx + m
            # ].to_numpy()  # The distance calculation requires they are in NumPy arrays.
            top_1_motif = T[j : j + m]
            # top_1_motif = T[j : j + m].to_numpy()
            top_1_motif_dist[i] = np.linalg.norm(
                stumpy.core.z_norm(query) - stumpy.core.z_norm(top_1_motif)
            )
            top_1_motif_point_after_cols = {
                f"top_1_motif_point_after_{ll+1}": top_1_motif_points_after[:, ll]
                for ll in range(l)
            }
            df_motif = pd.DataFrame(
                {
                    "top_1_motif_dist": top_1_motif_dist,
                    "top_1_motif_idx": top_1_motif_idx,
                    "top_1_motif_idx_delta": top_1_motif_idx_delta,
                    **top_1_motif_point_after_cols,
                }
            )

    return df_motif

def get_top_k_motifs(T, m, k=1, l=1, include_itself=False):
    # https://github.com/stumpy-dev/stumpy/discussions/1093#discussioncomment-14063985

    # print("get_top_k_motifs")
    top_k_motif_dist = np.full((len(T), k), np.nan)
    top_k_motif_idx = np.full((len(T), k), np.nan)
    top_k_motif_idx_delta = np.full((len(T), k), np.nan)
    top_k_motif_points_after = np.full((len(T), k, l), np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
            **top_k_motif_dist_cols,
            **top_k_motif_idx_cols,
            **top_k_motif_idx_delta_cols,
            **top_k_motif_point_after_cols,
        }
    )
    return df_motif


def compute_point_after_average(df, method="unweighted"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        motif_dist_cols = [c for c in df.columns if re.match(r"top_\d+_motif_dist", c)]
        # r: raw string literal
        # \d: a digit
        motif_ranks = [
            int(re.search(r"top_(\d+)_motif_dist", c).group(1)) for c in motif_dist_cols
        ]
        # group(1): the first capture group (the part in the first pair of parentheses)
        # print(f"Motif ranks detected: {motif_ranks}")
        # Detect all point_after indices
        point_after_cols = [c for c in df.columns if "point_after_" in c]
        # To ensure the points are in order
        point_after_indices = sorted(
            {int(re.search(r"point_after_(\d+)", c).group(1)) for c in point_after_cols}
        )
        # print(f"Motif point_after_indices detected: {point_after_indices}")
        # Result
        results = pd.DataFrame(index=df.index)
        if method == "weighted":
            for j in point_after_indices:
                # Weighted average numerator & denominator
                num = sum(
                    df[f"top_{k}_motif_dist"].fillna(0)
                    * df[f"top_{k}_motif_point_after_{j}"].fillna(0)
                    for k in motif_ranks
                    if f"top_{k}_motif_point_after_{j}" in df.columns
                )
                den = sum(
                    df[f"top_{k}_motif_dist"].fillna(0)
                    for k in motif_ranks
                    if f"top_{k}_motif_point_after_{j}" in df.columns
                )
                results[f"weighted_average_point_after_{j}"] = np.where(
                    den > 0, num / den, np.nan
                )
        elif method == "unweighted":
            for j in point_after_indices:
                # Unweighted average (only motifs where dist exists)
                values = []
                for k in motif_ranks:
                    col = f"top_{k}_motif_point_after_{j}"
                    if col in df.columns:
                        mask = df[f"top_{k}_motif_dist"].notna()
                        values.append(np.where(mask, df[col], np.nan))

                if values:
                    stacked = np.column_stack(values)
                    results[f"average_point_after_{j}"] = np.nanmean(stacked, axis=1)
        else:
            raise ValueError("Method must be 'weighted' or 'unweighted'.")
    return results