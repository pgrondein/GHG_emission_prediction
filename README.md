
# Green House Gas Emission prediction

![banniere2](https://github.com/pgrondein/GHG_emission_prediction/assets/113172845/9c75d319-2ca6-48fc-9c74-3f9f053aa465)

## Context

Goal 2050 carbon neutral city for the city of Seattle.

Consumption and emissions from non-residential buildings.

Estimation of the efficiency and development of a model for energy consumption of said buildings in order to prevent the need of expended measures.

## Data

Careful surveys were carried out in 2015 and 2016. The dataset is available [here](https://data.seattle.gov/dataset/2016-Building-Energy-Benchmarking/2bpz-gwpy) .

Data is from 2015 and 2016.

- **2015**: 3340 rows  and 47 features.
- **2016**: 3376 rows and 46 features.

## Features

Features selected for the model, apart from building identification features, are:

- **Building Type/Property type**: City of Seattle building type classification
- **Year built**: Year in which a property was constructed or underwent a complete renovation
- **Property Gross Floor Area Total**: Total building and parking gross floor area
- **Electricity/Natural Gas/Steam Use**: The annual amount of district electricity/natural gas/steam consumed by the property on-site
- **Source Energy Use**: The annual energy used to operate the property, including losses related to the production, transport and distribution of this energy.

The “targets” variables are as follows:

- **Site Energy Use**: The annual amount of energy consumed by the property from all sources of energy
- **Total Green House Gas Emissions**: The total amount of greenhouse gas emissions, including carbon dioxide, methane, and nitrous oxide gases released into the atmosphere as a result of energy consumption at the property, measured in metric tons of carbon dioxide equivalent.

## Exploratory Analysis

### Univariate Analysis

#### Type of Building

<img src="https://github.com/pgrondein/GHG_emission_prediction/assets/113172845/d26a71cb-4a6d-418a-b433-de82fab40e49" height="400">

The exploratory analysis on the distribution of building types reveals that only 50% of the buildings considered are non-residential. Residential building data is removed.

#### Year Built

<img src="https://github.com/pgrondein/GHG_emission_prediction/assets/113172845/2cdc417f-72bd-41b0-aa42-a9c95f2ec410" height="400">

#### Total Property Gross Floor Area

<img src="https://github.com/pgrondein/GHG_emission_prediction/assets/113172845/e11b9477-72f4-446f-8f45-5c4eda14d8ca" height="400">

#### Target 1 : Total Green House Gas Emissions

<img src="https://github.com/pgrondein/GHG_emission_prediction/assets/113172845/e95f3513-1323-4797-bd9e-4f6b13b434b0" height="400">

#### Target 2 : Site Energy Use 

<img src="https://github.com/pgrondein/GHG_emission_prediction/assets/113172845/e2ef019f-d0ab-4f46-9c21-aa0ff4203fae" height="400">

### Bivariate Analysis

![corr](https://github.com/pgrondein/GHG_emission_prediction/assets/113172845/204a9137-639b-4b66-818c-44bb4265746c)

Some quantitative features appear to be strongly correlated.

Let's perform statistical tests to check Pearson's coefficient values. Let's make the assumptions:

- H0: Independent variables if p-value > a%
- H1: Non-independent variables if p-value < a%

We will choose a = 5 by default.

Now let's calculate the p-values.

- The **TotalGHGEmissions** target and the **PropertyGFATotal** variable are correlated, with a p-value < 5%.
- The **SiteEnergyUse(kBtu)** target and the **PropertyGFATotal** variable are correlated, with a p-value < 5%.
- The **SiteEnergyUse(kBtu)** and **TotalGHGEmissions** targets are correlated, with a p-value < 5%.

The two selected targets seem correlated with at least one of the explanatory variables (PropertyGFATotal), which confirms that it is interesting to use the selected dataset to predict them.

# Data preparation

After separating data into **test** and **train** dataset for the model, they are separately cleaned (missing values, outliers, etc.). Then quantitative features are normalized, categorical features binarized.

# Modelization

## Tested Models

### Dummy Regressor

A very simple regression model to compare with the other more complex models.

### Linear Regression

A linear regression model is a model that seeks to establish a linear relationship between a so-called explained feature and one or more so-called explanatory features.

To **limit overfitting**, we can use a technique, **regularization**, which consists in simultaneously controlling the error and the complexity of the model. Two regularization modes are tested:

- **Ridge regularization**: regression model with a l2 regularization term
- **Lasso regularization**: regression model with an l1 regularization term

### Decision Tree

A **Decision Tree** is a Machine Learning algorithm for classifying data based on sequences of conditions. It is a nonlinear binary decision sequence model.

### RandomForest

**Random forests** or **random decision forests** is an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time.

## Metrics

In order to decide between the different models tested, the following metrics were used:

- **R²**: Coefficient of determination, square of the Pearson correlation, must be maximized.
- **MAE** (Mean Absolute Error) & MAPE (Mean Absolute Percentage Error): sum of absolute errors divided by the size of the sample. Should be minimized.
- **RMSE** (Root Mean Squared Error): Should be minimized.
- **Computation time**

# Conclusion

The best hyperparameters are determined by GridSearch.

|  | R²  | MAE | MAPE | RMSE | Computational Time |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `Dummy Regressor`  | 0.00  | >> 1 | >> 1 | >> 1 | <<< 0.001 s |
| `Linear Regression`  | 0.78  | 0.54 | 0.26 | 0.71 | <<< 0.001 s |
| `Regression Ridge`  | 0.82  | 0.55 | 0.25 | 0.71 | <<< 0.001 s |
| `Regression Lasso`  | 0.80  | 0.53 | 0.25 | 0.70 | 0.12 s |
| `Decision Tree`  | 0.94  | 0.16 | 0.04 | 0.40 | <<< 0.001 s |
| `Random Forest`  | 0.96  | 0.12 | 0.03 | 0.31 | 0.48 s |
