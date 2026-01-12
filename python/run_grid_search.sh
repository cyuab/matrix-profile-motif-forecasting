#!/bin/bash

# Stop execution if any command fails
set -e

echo "For electricity..."
echo "Starting Grid Search: electricity, include_motif_information=0..."
python grid_search_electricity.py \
    --include_covariates True False

echo "Starting Grid Search: electricity, include_motif_information=1..."
python grid_search_electricity.py \
    --include_covariates True False \
    --include_motif_information 1 \
    --no_points_after_motif 1 2 6 12 24\
    --do_normalization True False \
    --include_similarity True False

echo "Starting Grid Search: electricity, include_motif_information=2..."
python grid_search_electricity.py \
    --include_covariates True False \
    --include_motif_information 2 \
    --no_points_after_motif 1 2 6 12 24\
    --do_normalization True False \
    --include_similarity True False


echo "For traffic..."
echo "Starting Grid Search: traffic, include_motif_information=0..."
python grid_search_traffic.py \
    --include_covariates True False

echo "Starting Grid Search: traffic, include_motif_information=1..."
python grid_search_traffic.py \
    --include_covariates True False \
    --include_motif_information 1 \
    --no_points_after_motif 1 2 6 12 24\
    --do_normalization True False \
    --include_similarity True False

echo "Starting Grid Search: traffic, include_motif_information=2..."
python grid_search_traffic.py \
    --include_covariates True False \
    --include_motif_information 2 \
    --no_points_after_motif 1 2 6 12 24\
    --do_normalization True False \
    --include_similarity True False

# echo "For pemds7..."
# echo "Starting Grid Search: pemds7, include_motif_information=0..."
# python grid_search_pemds7.py \
#     --include_covariates True False

# echo "Starting Grid Search: pemds7, include_motif_information=1..."
# python grid_search_pemds7.py \
#     --include_covariates True False \
#     --include_motif_information 1 \
#     --no_points_after_motif 1 2 3 5 9 \
#     --do_normalization True False \
#     --include_similarity True False

# echo "Starting Grid Search: pemds7, include_motif_information=2..."
# python grid_search_pemds7.py \
#     --include_covariates True False \
#     --include_motif_information 2 \
#     --no_points_after_motif 1 2 3 5 9 \
#     --do_normalization True False \
#     --include_similarity True False

# echo "For rate_exchange..."
# echo "Starting Grid Search: rate_exchange, include_motif_information=0..."
# python grid_search_rate_exchange.py \
#     --include_covariates True False

# echo "Starting Grid Search: rate_exchange, include_motif_information=1..."
# python grid_search_rate_exchange.py \
#     --include_covariates True False \
#     --include_motif_information 1 \
#     --no_points_after_motif 1 2 6 12 24\
#     --do_normalization True False \
#     --include_similarity True False

# echo "Starting Grid Search: rate_exchange, include_motif_information=2..."
# python grid_search_rate_exchange.py \
#     --include_covariates True False \
#     --include_motif_information 2 \
#     --no_points_after_motif 1 2 6 12 24\
#     --do_normalization True False \
#     --include_similarity True False

echo "All grid searches completed successfully."