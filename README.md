# Leveraging Nearest Neighbors for Time Series Forecasting with Matrix Profile

## Notifications

## Installation
```
    conda create -n mpmf python=3.10 pandas numpy matplotlib seaborn scikit-learn xgboost 
    conda activate mpmf
    conda install -c conda-forge stumpy
    conda deactivate
    conda env remove -n mpmf
```
- "mpmf" stands for **m**atrix-**p**rofile-**m**otif-**f**orecasting (i.e., the repository name).

## Datasets
- Electricity [[sen2019think]](#sen2019think)
- Traﬃc [[sen2019think]](#sen2019think)
- PeMSD7 [[sen2019think]](#sen2019think)
- Exchange-Rate [[lai2018modeling]](#lai2018modeling)

## Project Structure
- GBRT_Univariate/: The original code ([Commit 9b14dc9](https://github.com/Daniela-Shereen/GBRT-for-TSF/tree/9b14dc957cb2f33fc4a04566d9c140dc2b2a3014), accessed on 2025-12-23) from [[elsayed2021we]](#elsayed2021we) with little modifications such that it can be run in our environment and the result is reproducible by setting the random seed.
- environment_information.ipynb: Check environment information.
  
## Corresponding Paper
### Figures/Tables in the Paper

## References
Sorted by year
- <a id="lai2018modeling"></a>[lai2018modeling] "Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks" [(code)](https://github.com/fbadine/LSTNet)
- <a id="sen2019think"></a>[sen2019think] "Think Globally, Act Locally: A Deep Neural Network Approach to High-Dimensional Time Series Forecasting" [(code)](https://github.com/rajatsen91/deepglo/tree/525578d4bd50a9d71abbf8b51f0ce9987f449db8?tab=readme-ov-file)
- <a id="elsayed2021we"></a>[elsayed2021we] "Do We Really Need Deep Learning Models for Time Series Forecasting?" [(code)](https://github.com/Daniela-Shereen/GBRT-for-TSF)
  