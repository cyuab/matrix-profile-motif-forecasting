# How to run the code?

- The code is finally tested on my notebook computer (Apple M1 Pro chip, 16 GB memory, namely daikon).
- It has also been tested on other machine (rwcpu9 (10), CSE Compute).

## Input parameters

- `include_motif_information`
  - Baseline 
    - `0`: GBRT (Original)
  - 1-NN
    - `1`: GBRT-NN: With the immediate subsequence (of length `l`) of `no_points_after_motif` of the NN
      - `2`: Pairwise (trend): Same but use pairwise change instead of raw values of the immediate subsequence
    - `3`: GBRT-NN<sup>+</sup>: With the target value of the last point of the NN on top of `1`
      - `4`: Pairwise
    - `5`: GBRT-NN<sup>++</sup>: With the target value of all the points of the NN on top of `1`
      - `6`: Pairwise
    - Covariates 
      - `7`: GBRT-NNC: With the covariates of the last point of the NN on top of `1` 
        - `8`: Pairwise
      - `9`: GBRT-NNC<sup>+</sup>: With the covariates of the last point of the NN on top of `3` 
        - `10`: Pairwise
      - `11`: GBRT-NNC<sup>++</sup>: With the covariates of the last point of the NN on top of `5` 
        - `12`: Pairwise
  - k-NN
    - `13`: GBRT-kNN version of `1`
    - `15`: GBRT-kNN version of `3`
    - `17`: GBRT-kNN version of `5`
    - Covariates
      - `19`: GBRT-kNN version of `7`
      - `21`: GBRT-kNN version of `9`
      - `23`: GBRT-kNN version of `11`
  - Remarks
    - “-S”: With similarity
    - Odd numbers: Raw value; Even numbers: Pairwise (trend)

## Test on my notebook computer

```
  # Comment/uncomment to execute the desired experiments 
  $ ./run_grid_search.sh
  # If permission denied
  $ chmod +x run_grid_search.sh
```

## Test on rwcpu9 (10)

```
  # Only use the normal version instead of the GPU version of stumpy
  $ export NUMBA_DISABLE_CUDA=1
  # Use tmux to detach a task from the terminal
  # Four datasets: electricity, pemds7, rate_exchange, traffic
  $ tmux new -s electricity
  $ ./run_grid_search_electricity.sh
  $ tmux new -s pemds7
  $ ./run_grid_search_pemds7.sh
  $ tmux new -s rate_exchange
  $ ./run_grid_search_rate_exchange.sh
  $ tmux new -s traffic
  $ ./run_grid_search_traffic.sh

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

### MISC
```
# List all users
# https://unix.stackexchange.com/questions/617652/how-can-i-list-all-currently-logged-in-users
$ ps -eo user,uid | awk 'NR>1 && $2 >= 1000 && ++seen[$2]==1{print $1}'

# Check Current Tasks
$ top
# Press M to sort by Memory usage or P to sort by CPU usage.
# Pree q to exit.

# Check GPU information
$ nvidia-smi

# Check # CPU
$ nproc
```
- environment
```
$ conda install -c conda-forge matplotlib -y
$ conda install -c conda-forge seaborn -y
```

## Test on CSE Compute, HKUST

- On [COMPUTE service](https://compute.cse.ust.hk/), [Computing Facilities, CS System, Department of CSE, HKUST.](https://cssystem.cse.ust.hk/Facilities/index.html)  

- Dockerfile
    ```
    FROM python:3.10-slim
    RUN pip install --no-cache-dir numpy pandas scikit-learn xgboost stumpy
    ```
- Usage examples
  - Using *.py directly
    ```
    HOME=/project/kdd/cyuab2/matrix-profile-motif-forecasting/; cd ~; pwd; cd python; pwd; python grid_search_pemds7.py --include_covariates True False

    HOME=/project/kdd/cyuab2/matrix-profile-motif-forecasting/; cd ~; pwd; cd python; pwd; python grid_search_pemds7.py --include_covariates True False --include_motif_information 1 --no_points_after_motif 1 5 9 --do_normalization True False --include_similarity True False
    ```
  - Or use *.sh instead
    ```
    HOME=/project/kdd/cyuab2/matrix-profile-motif-forecasting/; cd ~; pwd; cd python; pwd; echo "--- CPU Model ---"; grep -m 1 'model name' /proc/cpuinfo; ./run_grid_search_pemds7.sh
    ```
    - `run_grid_search_*.py` just run a part of of `run_grid_search.py`.