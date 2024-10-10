# TSFM for Load-forecasting

## Project Description
Accurate short-term energy load forecasting (STLF) is essential for decarbonizing the buildings sector, which accounts for about one-third of global energy consumption and greenhouse gas (GHG) emissions. Accurate forecasts enable efficient energy management 
by predicting future energy needs, aligning supply with demand, reducing reliance on fossil fuels, integrating renewable energy sources, and optimizing energy use within buildings. While various methods—ranging from statistical to machine learning—have been proposed, 
their evaluation often focuses on a limited number of buildings, hindering scalability and generalizability. To address these scalability limitations, transfer learning has emerged as a promising solution by leveraging data from similar buildings to enhance forecasting 
accuracy for target buildings while reducing data requirements. In recent years, Time Series Foundation Models (TSFMs) have been developed for universal time series analysis, pre-trained on vast time series corpus from various domains, including energy, and across different time granularities, 
enabling them to learn generalizable temporal patterns. Unlike transfer learning, TSFMs adopt a holistic approach, allowing them to make accurate forecasts on new, unseen time series without requiring any retraining or fine-tuning. This makes them suitable for implementing 
STLF for any building using a single model. However, a comprehensive assessment of their applicability and performance for large-scale STLF is still lacking, which is necessary for their adoption. 

To address this gap, we analyze the performance of four state-of-the-art open-source TSFMs -- [Chronos](https://github.com/amazon-science/chronos-forecasting), [Lag-Llama](https://github.com/time-series-foundation-models/lag-llama.git), [Moirai](https://github.com/SalesforceAIResearch/uni2ts), and [TimesFM](https://github.com/google-research/timesfm) -- for STLF in both commercial and residential buildings.

In this repository, we present a comprehensive evaluation of four TSFMs for zero-shot performance on a large-scale dataset comprising over 
1,900 real-world residential and commercial buildings from BuildingsBench Evaluation Platform.
#### STLF
The univariate short-term load forecasting (STLF) problem can be defined
as follows: Given *H* past load values *x*<sub>*t* − *H* : *t*</sub>, we
aim to predict the conditional distribution of *T* unobserved future
load values *y*<sub>*t* + 1 : *t* + *T*</sub>. For our analysis, we
considered the STLF problem of forecasting the next day’s load values (T
= 24) using data from the past week (H = 168).

## Dataset
[BuildingsBench](https://github.com/NREL/BuildingsBench.git) is an open-source evaluation platform designed to benchmark load forecasting models. It consists of the Buildings-900K dataset, a large-scale dataset of hourly energy time series from 900K simulated buildings in the USA,
and a test dataset collected from over 1,900 real residential and commercial buildings across the world. Additionally, the BuildingsBench platform presents a transformer-based foundation model which was pre-trained using the Buildings-900K dataset. 
They compare its performance with various state-of-the-art machine learning algorithms under zero-shot and transfer learning settings. In this project, we leverage the same test buildings dataset as used in the BuildingsBench for evaluation and also compare the zero-shot performance of the four selected TSFMs with all other models included in BuildingsBench.
The Datset can be downloaded from the BuildingsBench [repository](https://github.com/NREL/BuildingsBench)  or can be accessed directly from [Data](https://data.openei.org/s3_viewer?bucket=oedi-data-lake&prefix=buildings-bench).
#### Sliding Window Extraction
We began by extracting sliding windows for each building and year separately. Specifically, we employed an 8-day sliding window comprising a 192-hour load sub-sequence. The initial 7 days (168 hourly energy meter readings) 
served as context to forecast the subsequent 24-hour readings of the $8^{th}$ day, similar to the BuildingsBench.

| **Dataset** | **#Buildings** | **Years** | **#Windows** |
|----------------------|--------------------------:|:--------------------:|------------------------:|
| Commercial Buildings |                          |                    |                        |
| BDG-2                | 611                      | 2016-17            | 192,962                |
| Buildings-900K-test  | 565                      | 2018               | 101,135                |
| Electricity          | 359                      | 2011-14            | 207,461                |
| Residential Buildings|                          |                    |                        |
| LCL                  | 713                      | 2012-13            | 190,940                |
| IDEAL                | 219                      | 2016-18            | 44,296                 |
| Borealis             | 15                       | 2011               | 1,772                  |
| SMART                | 5                        | 2014-16            | 3,613                  |
| Sceaux               | 1                        | 2007-10            | 1,741                  |

## Getting Started
We recommend using [Anaconda](https://www.anaconda.com/download) to run the experiments. Create the separate conda environment using the (modelname)_environment.yml found
under each model directory in the Notebooks. 
```
conda env create -f <modelname>_environment.yml
```
For TimesFM, kindly follow the respective [READEME.md](https://github.com/google-research/timesfm/blob/master/README.md) for the installation. For more information about each model, kindly look in 
their Github repositories.

## Benchmarking
Comparison of models performance using median NRMSE. First four category results are adopted
from BuildingsBench. Best model under each category is in Italics. (Cat.: Category)

| **Models**                                           | **Commercial**             | **Residential**            |
|------------------------------------------------------|----------------------------|----------------------------|
| Cat. - **Not pretrained + Not fine-tuned**           |                            |                            |
| Persistence Ensemble                                 | 16.80                      | *78.54*                    |
| Previous Day Persistence                             | *16.54*                    | 98.35                      |
| Previous Week Persistence                            | 18.93                      | 100.20                     |
| Cat. - **Not pretrained + Fine-tuned**               |                            |                            |
| Linear regression                                    | 25.18                      | 89.98                      |
| DLinear                                              | 23.41                      | 87.89                      |
| RNN (Gaussian)                                       | 41.79                      | 96.75                      |
| LightGBM                                             | *16.02*                    | *80.07*                    |
| Transformer-L (Tokens)                               | 50.12                      | 105.65                     |
| Transformer-L (Gaussian)                             | 37.21                      | 92.99                      |
| Cat. - **Pretrained + Not fine-tuned**               |                            |                            |
| Transformer-L (Tokens)                               | 14.20                      | 94.11                      |
| Transformer-L (Gaussian)                             | *13.03*                    | *79.43*                    |
| Cat. - **Pretrained + Fine-tuned**                   |                            |                            |
| Transformer-L (Tokens)                               | 14.07                      | 94.53                      |
| Transformer-L (Gaussian)                             |*12.96*                     | *77.20*                    |
| Cat. - **Trained From Scratch Transformer**          |                            |                            |
| PatchTST                                             | *19.25*                    | 88.63                      |
| Temporal Fusion Transformer(TFT)                     | 21.83                      | *85.27*                    |
|Cat. - **Time Series Foundation Models (Zero-shot)**  |                            |                            |
| TimesFM (200M)                                       | 20.01                      | *86.54*                    |
| Lag-llama (2.45M)                                    | 28.97                      | 102.88                     |
| Moirai (14M)                                         | 25.0                       | 94.98                      |
| Chronos (46M)                                        | *13.56*                    | 87.85                      |

## Directory structure
```
.
└── Notebooks/
    ├── Lag-Llama         <- Zero-shot using Lag-Llama
    ├── TimesFM           <- Zero-shot using TimesFM
    ├── Moirai            <- Zero-shot using Moirai
    └── Chronos           <- Zero-shot using Chronos and Error analysis for commercial and residential buildings datasets 
```
