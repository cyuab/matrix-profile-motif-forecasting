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
They will be uploaded to zenodo.
- Electricity [sen2019think]
- Traﬃc [sen2019think]
- PeMSD7 [sen2019think]
- Exchange-Rate [lai2018modeling]
- Solar-Energy [lai2018modeling]

### References
- [sen2019think] "Think Globally, Act Locally: A Deep Neural Network Approach to High-Dimensional Time Series Forecasting" [(code)](https://github.com/rajatsen91/deepglo/tree/525578d4bd50a9d71abbf8b51f0ce9987f449db8?tab=readme-ov-file)
- [lai2018modeling] "Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks" [(code)](https://github.com/fbadine/LSTNet)

## Project Structure
- GBRT_Univariate: The original code from "Do We Really Need Deep Learning Models for Time Series Forecasting?" with little modifications such that it can be run in our environment.
- environment_information.ipynb: Environment information.
## Corresponding Paper
### Figures/Tables in the Paper
## Contacts