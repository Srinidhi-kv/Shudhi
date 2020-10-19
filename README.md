# Shudhi: A must have Data science module

## About

We are Data Science graduate students at Columbia University. We regularly analyse various datasets and the thing that bothers us the most is that the time & effort invested in the Processing and exploration phase is too much. This phase takes the most time in the Data Analysis journey (>50% time). It is extremely important to understand the dataset thoroughly before even thinking about ML models, but, most steps in the phase can be automated to save time and effort.

## Authors

Aishwarya Srinivasan, Data Science, Columbia University: https://www.linkedin.com/in/aishwarya-srinivasan/

Srinidhi KV, Data Science, Columbia University: https://www.linkedin.com/in/srinidhi-kv/

## Shudhi

Shudhi stands for purification in many Indian languages. Here, we aim to "purify" data so that it is easy to comprehend and ML models can be built on top of them with minimum effort.

To use the module, you need to download the "shudhi.py" file, paste it in the lib/site-packages/ or paste in the same directory as your python notebook/file, and use the import statement as follows:

```{python}
from shudhi import shudhi_describe, shudhi_transform
```

### 1. Shudhi Describe

Shudhi Describe: aids understanding by thoroughly describing a dataset: outputs summary statistics, Univariate and Bivariate plots(with the Target) and a fine correlation plot.

#### Function call
```{python}
shudhi_describe(df_train, cols=[None], empty_missing=False, plot=True, target=None)
```
|Parameter| Description|
|:---:|:---:|
|df_train | Training dataframe|
|cols| list of columns|
|empty_missing | ‘True’ to consider empty strings as np.nan |
|plot | ‘False’ to turf off plots |
|target |  Target column name |

#### (Please go through the shudhi_describe notebook/html to see a demo)

### 2. Shudhi Transform
Shudhi Transform: transforms the data by doing the most used pre processing methods: Missing value treatment, Outlier Treatment, Feature Scaling, Type Conversion and One hot encoding, on both train and test datasets.

#### Function call

```{python}
shudhi_transform(df_train, df_test= None, cols= [None], missing_strategy=None, empty_missing=False, missing_override=False, scale_strategy=None, outlier_strategy=None, one_hot=False, convert=False)
```
|Parameter| Description|
|:---:|:---:|
|df_train | Training dataframe|
|df_test | Test Dataframe|
|cols| list of columns|
|missing_strategy|"remove"/"mean"/"median"/"mode" (imputes missing values)|
|empty_missing | ‘True’ to consider empty strings as np.nan |
|missing_overrideot | 'True' to override the 10% rule |
|scale_strategy | "std"/"min_max"/"robust"/"max_abs"/"norm" (Scales data accordingly) |
|outlier_strategy|"remove"/"min_max"/"mean (imputes outliers accordingly)|
|one_hot|'True' to perform one hot encoding|
|convert|'True' to convert dtypes|


#### (Please go through the shudhi_transform notebook/html to see a demo)

## Feedback Form

https://goo.gl/forms/E2kAqNsauIjBEVi33
