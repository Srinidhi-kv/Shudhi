# Shudhi: A must have Data science module

## Introduction

We are Data Science graduate students at Columbia University. We regularly analyse various datasets and the thing that bothers us the most is that the time & effort invested in the Processing and exploration phase is too much. This phase takes the most time in the Data Analysis journey (>50% time). It is extremely important to understand the dataset thoroughly before even thinking about ML models, but, most steps in the phase can be automated to save time and effort.

## Shudhi

Shudhi stands for purification in many Indian languages. Here, we aim to "purify" data so that it is easy to comprehend and ML models can be built on top of them with minimum effort. The module is divided into two sub-modules:

### Shudhi Describe

a. Shudhi Describe: aids understanding by thoroughly describing a data: outputs summary statistics, Univariate and Bivariate plots(with the Target) and a nice correlation plot.

#### Function call
```{python}
shudhi_describe(df_train, cols= [None], empty_missing= False, plot=True, target= None)
```

#### (Please go through the shudhi_describe notebook/pdf to see practical implementation)

### Shudhi Transform
b. Shudhi Transform: transforms the data by doing the most used pre processing methods: Missing value treatment, Outlier Treatment, Feature Scaling, Type Conversion and One hot encoding, on both train and test datasets.

#### Function call

```{python}
shudhi_transform(df_train, df_test= None, cols= [None], missing_strategy=None, empty_missing= False, missing_override=False, scale_strategy= None, outlier_strategy= None, one_hot= False, convert= False, imbalance_strategy= False)
```
#### (Please go through the shudhi_transform notebook/html to see practical implementation)

## Feedback Form

https://goo.gl/forms/E2kAqNsauIjBEVi33
