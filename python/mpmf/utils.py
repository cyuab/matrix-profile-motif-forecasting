import numpy as np
import pandas as pd
import stumpy
from stumpy import config, core
import re
import warnings
# import sys, os
from numba import njit, prange
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

@njit(cache=True)
def _compute_rolling_stats(T, m):
    n = len(T)
    means = np.full(n, np.nan)
    stds = np.full(n, np.nan)
    
    sum_x = 0.0
    sum_x2 = 0.0
    
    for i in range(m):
        val = T[i]
        sum_x += val
        sum_x2 += val * val
        
    means[0] = sum_x / m
    var = (sum_x2 / m) - (means[0] * means[0])
    if var < 1e-14: var = 0.0
    stds[0] = np.sqrt(var)
    
    for i in range(1, n - m + 1):
        prev_val = T[i-1]
        new_val = T[i+m-1]
        
        sum_x = sum_x - prev_val + new_val
        sum_x2 = sum_x2 - (prev_val * prev_val) + (new_val * new_val)
        
        mu = sum_x / m
        means[i] = mu
        var = (sum_x2 / m) - (mu * mu)
        if var < 1e-14: var = 0.0
        stds[i] = np.sqrt(var)
        
    return means, stds

@njit(parallel=True)
def _find_top_k_motifs_numba_core(T, m, k, excl_zone, means, stds, start_idx_loop, q_offset):
    n = len(T)
    out_dist = np.full((n, k), np.inf)
    out_idx = np.full((n, k), np.nan)
    out_delta = np.full((n, k), np.nan)
    
    for i in prange(start_idx_loop, n):
        q_idx = i - q_offset
        if q_idx < 0 or q_idx >= n - m + 1: continue
        
        mu_q = means[q_idx]
        sigma_q = stds[q_idx]
        if sigma_q < 1e-14: continue
        
        limit = q_idx - excl_zone
        if limit < 0: continue
        
        best_dists = np.full(k, np.inf)
        best_idxs = np.full(k, np.nan)
        
        for j in range(limit + 1):
             mu_c = means[j]
             sigma_c = stds[j]
             if sigma_c < 1e-14: continue
             
             dot = 0.0
             for off in range(m):
                 dot += T[q_idx + off] * T[j + off]
             
             denom = m * sigma_q * sigma_c
             if denom == 0: continue
             cov = dot - m * mu_q * mu_c
             corr = cov / denom
             if corr > 1.0: corr = 1.0
             elif corr < -1.0: corr = -1.0
             dist = np.sqrt(2 * m * (1.0 - corr))
             
             if dist < best_dists[k-1]:
                 pos = k - 1
                 while pos > 0 and (best_dists[pos-1] > dist or np.isnan(best_dists[pos-1])):
                     best_dists[pos] = best_dists[pos-1]
                     best_idxs[pos] = best_idxs[pos-1]
                     pos -= 1
                 best_dists[pos] = dist
                 best_idxs[pos] = j
                 
        out_dist[i] = best_dists
        out_idx[i] = best_idxs
        for kk in range(k):
            if not np.isnan(best_idxs[kk]):
                out_delta[i, kk] = q_idx - best_idxs[kk]
                
    return out_dist, out_idx, out_delta

def get_top_k_motifs_numba(T, m, k=1, l=1, include_itself=False):
    if isinstance(T, pd.Series):
        T = T.values
    T = T.astype(np.float64)
    
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    means, stds = _compute_rolling_stats(T, m)
    
    if include_itself:
        start_idx_loop = m - 1
        q_offset = m - 1
    else:
        start_idx_loop = m
        q_offset = m
        
    out_dist, out_idx, out_delta = _find_top_k_motifs_numba_core(
        T, int(m), int(k), int(excl_zone), means, stds, int(start_idx_loop), int(q_offset)
    )
    
    # Convert inf to nan to match behavior of original function
    out_dist[np.isinf(out_dist)] = np.nan
    
    data = {}
    for kk in range(k):
        data[f"top_{kk+1}_motif_dist"] = out_dist[:, kk]
        data[f"top_{kk+1}_motif_idx"] = out_idx[:, kk]
        data[f"top_{kk+1}_motif_idx_delta"] = out_delta[:, kk]
        
    # Extract points after (Python loop is fast enough for linear extraction)
    for kk in range(k):
        for ll in range(l):
            col_name = f"top_{kk+1}_motif_point_after_{ll+1}"
            col_vals = np.full(len(T), np.nan)
            
            # Vectorized extraction if possible or list comprehension
            idxs = out_idx[:, kk]
            
            # Mask for valid indices
            mask = ~np.isnan(idxs)
            valid_indices = np.where(mask)[0]
            
            for i in valid_indices:
                idx_neighbor = int(idxs[i])
                target = idx_neighbor + m + ll
                if target < len(T):
                    col_vals[i] = T[target]
            
            data[col_name] = col_vals
            
    return pd.DataFrame(data)

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

def get_top_k_motifs_refined(T, m, k=1, l=1):
    n = len(T)
    excl_zone = int(np.ceil(m / 4)) # Standard Stumpy exclusion zone

    dists_out = np.full((n, k), np.nan)
    idxs_out = np.full((n, k), np.nan)
    points_after_out = np.full((n, k, l), np.nan)

    for i in range(m, n):
        query = T[i-m:i]
        # 1. Compute distance profile for the past T[:i]
        # This is the fast Numba-optimized part of Stumpy
        dist_prof = stumpy.core.mass(query, T)
        
        # 2. Mask the "future" and the current query's immediate vicinity
        # We only want matches that happened in the past
        dist_prof[i - m - excl_zone:] = np.inf 
        
        # 3. Iteratively find k non-overlapping matches
        for kk in range(k):
            best_match_idx = np.argmin(dist_prof)
            min_dist = dist_prof[best_match_idx]
            
            # If we hit infinity, there are no more valid matches in the past
            if min_dist == np.inf:
                break
                
            dists_out[i, kk] = min_dist
            idxs_out[i, kk] = best_match_idx
            
            # Extract points after
            after_start = int(best_match_idx + m)
            after_end = after_start + l
            p = T[after_start : min(after_end, n)]
            if len(p) > 0:
                points_after_out[i, kk, :len(p)] = p

            # 4. Apply exclusion zone around this match to prevent overlaps
            start_mask = max(0, best_match_idx - excl_zone)
            end_mask = min(n, best_match_idx + excl_zone + 1)
            dist_prof[start_mask:end_mask] = np.inf

    # ... (DataFrame construction logic same as before) ...
    return build_df(dists_out, idxs_out, points_after_out, n, k, l)

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

                start_slice_idx = matches[kk, 1] + m
                end_slice_idx = matches[kk, 1] + m + l
                # Slice T safely; NumPy handles out-of-bound end index by truncating
                points = T[start_slice_idx:end_slice_idx]
                
                # Assign what we found. If points is shorter than l, only fill the beginning.
                if len(points) > 0:
                    top_k_motif_points_after[i, kk, :len(points)] = points
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