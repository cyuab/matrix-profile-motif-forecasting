#!/bin/bash

# Stop execution if any command fails
set -e

# ================= Grid Search: pemds7 (PARALLEL) =================
# SAFETY: Limit each job to 8 threads so 5 jobs x 8 threads = 40 cores (fits in your 48)
export NUMBA_NUM_THREADS=8
export OMP_NUM_THREADS=8

echo "For pemds7..."

echo "Starting Grid Search: pemds7, include_motif_information=0..."
python grid_search_pemds7.py \
    --include_covariates True False &  # <--- Added &

echo "Starting Grid Search: pemds7, include_motif_information=1..."
python grid_search_pemds7.py \
    --include_covariates True False \
    --include_motif_information 1 \
    --no_points_after_motif 1 2 3 5 9 \
    --do_normalization True False \
    --include_similarity True False &  # <--- Added &

echo "Starting Grid Search: pemds7, include_motif_information=2..."
python grid_search_pemds7.py \
    --include_covariates True False \
    --include_motif_information 2 \
    --no_points_after_motif 1 2 3 5 9\
    --do_normalization True False \
    --include_similarity True False &  # <--- Added &

echo "Starting Grid Search: pemds7, include_motif_information=3, 5"
python grid_search_pemds7.py \
    --include_covariates True False \
    --include_motif_information 3 5 \
    --no_points_after_motif 9 \
    --include_similarity True False &  # <--- Added &

echo "Starting Grid Search: pemds7, include_motif_information=7, 9, 11..."
python grid_search_pemds7.py \
    --include_covariates True \
    --include_motif_information 7 9 11 \
    --no_points_after_motif 9 \
    --include_similarity True False &  # <--- Added &

# ==========================================================

# Wait for all background jobs to finish
wait

echo "All grid searches completed successfully."