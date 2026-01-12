# How to run the code?

- The code is finally tested on my notebook computer (Apple M1 Pro chip, 16 GB memory).
- It has also been tested on other machines.

## Test on my notebook computer

```
  # Comment/uncomment to execute the desired experiments 
  $ ./run_grid_search.sh
  # If permission denied
  $ chmod +x run_grid_search.sh
```

## Test on rwcpu10.cse.ust.hk

```
  # Only use the normal version instead of the GPU version of stumpy
  $ export NUMBA_DISABLE_CUDA=1
  # Use tmux to detach a task from the terminal
  $ tmux new -s electricity
  $ ./run_grid_search_electricity.sh

  # List tmux sessions
  $ tmux ls
  # Press Ctrl+B then D to exit
  # Attach
  $ tmux attach -t electricity
  # Kill this session
  $ tmux kill-session -t electricity
  # Kill all
  $ tmux kill-server
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
  - Using *.py directly (Only some examples are shown)
    ```
    HOME=/project/kdd/cyuab2/matrix-profile-motif-forecasting/; cd ~; pwd; cd python; pwd; python grid_search_pemds7.py --include_covariates True False

    HOME=/project/kdd/cyuab2/matrix-profile-motif-forecasting/; cd ~; pwd; cd python; pwd; python grid_search_pemds7.py --include_covariates True False --include_motif_information 1 --no_points_after_motif 1 5 9 --do_normalization True False --include_similarity True False

    HOME=/project/kdd/cyuab2/matrix-profile-motif-forecasting/; cd ~; pwd; cd python; pwd; python grid_search_pemds7.py --include_covariates True False --include_motif_information 2 --no_points_after_motif 1 5 9 --do_normalization True False --include_similarity True False
    ```
  - Or use *.sh instead
    ```
    HOME=/project/kdd/cyuab2/matrix-profile-motif-forecasting/; cd ~; pwd; cd python; pwd; ./run_grid_search_electricity.sh

    HOME=/project/kdd/cyuab2/matrix-profile-motif-forecasting/; cd ~; pwd; cd python; pwd; ./run_grid_search_traffic.sh

    HOME=/project/kdd/cyuab2/matrix-profile-motif-forecasting/; cd ~; pwd; cd python; pwd; ./run_grid_search_pemds7.sh

    HOME=/project/kdd/cyuab2/matrix-profile-motif-forecasting/; cd ~; pwd; cd python; pwd; ./run_grid_search_rate_exchange.sh
    ```