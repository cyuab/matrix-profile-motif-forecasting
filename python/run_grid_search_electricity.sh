#!/bin/bash

# Stop execution if any command fails
set -e

# ================= Grid Search: electricity =================
echo "For electricity..."

echo "Starting Grid Search: electricity, include_motif_information=0..."
python grid_search_electricity.py \
    --include_covariates True False

echo "Starting Grid Search: electricity, include_motif_information=1..."
python grid_search_electricity.py \
    --include_covariates True False \
    --include_motif_information 1 \
    --no_points_after_motif 1 2 6 12 24 \
    --do_normalization True False \
    --include_similarity True False

echo "Starting Grid Search: electricity, include_motif_information=2..."
python grid_search_electricity.py \
    --include_covariates True False \
    --include_motif_information 2 \
    --no_points_after_motif 1 2 6 12 24 \
    --do_normalization True False \
    --include_similarity True False

echo "Starting Grid Search: electricity, include_motif_information=3, 5"
python grid_search_electricity.py \
    --include_covariates True False \
    --include_motif_information 3 5 \
    --no_points_after_motif 24 \
    --include_similarity True False

echo "Starting Grid Search: electricity, include_motif_information=7, 9, 11..."
python grid_search_electricity.py \
    --include_covariates True \
    --include_motif_information 7 9 11 \
    --no_points_after_motif 24 \
    --include_similarity True False
# ==========================================================

echo "All grid searches completed successfully."