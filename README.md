# Leveraging Motifs for Time Series Forecasting with Matrix Profile

## Notifications


## Installation
```
    # mpmf stands for Matrix Profile Motif Forecasting (i.e., the repository name).
    conda create -n mpmf python=3.10 pandas numpy matplotlib seaborn scikit-learn xgboost 
    conda activate mpmf
    conda install -c conda-forge stumpy
    conda deactivate
    conda env remove -n mpmf
```
- We employ the code of "Do We Really Need Deep Learning Models for Time Series Forecasting?" paper.
  - [Commit 9b14dc9](https://github.com/Daniela-Shereen/GBRT-for-TSF/tree/9b14dc957cb2f33fc4a04566d9c140dc2b2a3014) (Accessed on 2025-12-23)

## Datasets
- Electricity [sen2019think]
- Traﬃc [sen2019think]
- PeMSD7 [sen2019think]
- Exchange-Rate [lai2018modeling]
- Solar-Energy [lai2018modeling]

### References
- [sen2019think] "Think Globally, Act Locally: A Deep Neural Network Approach to High-Dimensional Time Series Forecasting" [(code)](https://github.com/rajatsen91/deepglo/tree/525578d4bd50a9d71abbf8b51f0ce9987f449db8?tab=readme-ov-file)
- [lai2018modeling] "Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks" [(code)](https://github.com/fbadine/LSTNet)

## Project Structure
## Corresponding Paper
### Figures/Tables in the Paper
## Contacts

---
---
---
# GBRT-for-TSF

This read me file describes the steps to run the code for "Do We Really Need Deep Learning Models for 
Time Series Forecasting?" paper.

## Requirements*
* 1- numpy
* 2- pandas
* 3- XGBoost python library 
* 4- scikit-learn


## Datasets

Download the datasets and place them in the respective "Uni-" and/or "Multivariate" folders in the Data folder.

## Runing files

To run any of the scripts it is required to check the "data_path" of the required file, then running command is 

 - python file_name e.g. "python xgboostWB_electricity.py"
