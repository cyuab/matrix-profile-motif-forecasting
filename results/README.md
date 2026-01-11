# How to run the code?

- The code is finally tested on a notebook computer (Apple M1 Pro chip, 16 GB memory).
- It has also been tested on CSE Compute, HKUST.

## Test on a notebook computer

```
    # Comment/uncomment to execute the desired experiments 
    $ ./run_grid_search.sh
    # If permission denied
    $ chmod +x run_grid_search.sh
```

## Test on CSE Compute, HKUST

- On [COMPUTE service](https://compute.cse.ust.hk/), [Computing Facilities, CS System, Department of CSE, HKUST.](https://cssystem.cse.ust.hk/Facilities/index.html)  

- Dockerfile
    ```
    FROM python:3.10-slim
    RUN pip install --no-cache-dir numpy pandas scikit-learn xgboost stumpy
    ```
- Usage examples
  - Environment setting
    ```
    HOME=/project/kdd/cyuab2/matrix-profile-motif-forecasting/; cd ~; pwd; cd python; pwd; python grid_search_pemds7.py 
    ```
  ```
  # Job Name: Starting Grid Search: pemds7, include_motif_information=0...
  python grid_search_pemds7.py --include_covariates True False

  # Job Name: Starting Grid Search: pemds7, include_motif_information=1...
  python grid_search_pemds7.py --include_covariates True False --include_motif_information 1 --no_points_after_motif 1 5 9 --do_normalization True False --include_similarity True False

  # Job Name: Starting Grid Search: pemds7, include_motif_information=2...
  python grid_search_pemds7.py --include_covariates True False --include_motif_information 2 --no_points_after_motif 1 5 9 --do_normalization True False --include_similarity True False
  ```