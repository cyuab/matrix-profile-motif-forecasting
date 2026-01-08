- Dockerfile
    ```
    FROM python:3.10-slim
    RUN pip install --no-cache-dir numpy pandas scikit-learn xgboost stumpy
    ```
- For grid_search_pemds7.py
  - On [COMPUTE service](https://compute.cse.ust.hk/), [Computing Facilities, CS System, Department of CSE, HKUST.](https://cssystem.cse.ust.hk/Facilities/index.html)  
    ```
    HOME=/project/kdd/cyuab2/matrix-profile-motif-forecasting/; cd ~; pwd; cd python; pwd; python grid_search_pemds7.py 
    ```
```
# Job Name: grid_search_pemds7_results_[True False]_[True]_[0]_[1]_[1]
python grid_search_pemds7.py --include_covariates True False --include_itself True --include_motif_information 0 --k_motifs 1 --no_points_after_motif 1

# Job Name: grid_search_pemds7_results_[True False]_[True]_[1]_[1]_[1 5 9]
python grid_search_pemds7.py --include_covariates True False --include_itself True --include_motif_information 1 --k_motifs 1 --no_points_after_motif 1 5 9

# Job Name: grid_search_pemds7_results_[True False]_[True]_[2]_[2 3]_[1 5 9]
python grid_search_pemds7.py --include_covariates True False --include_itself True --include_motif_information 2 --k_motifs 2 3 --no_points_after_motif 1 5 9

# Job Name: grid_search_pemds7_results_[True False]_[True]_[3]_[2 3]_[1 5 9]
python grid_search_pemds7.py --include_covariates True False --include_itself True --include_motif_information 3 --k_motifs 2 3 --no_points_after_motif 1 5 9

# Job Name: grid_search_pemds7_results_[True False]_[True]_[4]_[2 3]_[1 5 9]
python grid_search_pemds7.py --include_covariates True False --include_itself True --include_motif_information 4 --k_motifs 2 3 --no_points_after_motif 1 5 9
```
- rate_exchange
- traffic
- electricity