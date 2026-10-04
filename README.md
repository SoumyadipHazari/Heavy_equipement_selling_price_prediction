# Heavy Equipment Selling Price Prediction

### MLP Project 2026 T2

A classical Machine Learning project for predicting the selling price of heavy equipment using structured operational, transactional, and technical data.

The project focuses on understanding the dataset through statistical analysis and exploratory data analysis, engineering meaningful features, comparing multiple regression algorithms, and optimizing the final model using **Root Mean Squared Logarithmic Error (RMSLE)**.

---

## Project Overview

The dataset contains information related to heavy industrial equipment transactions, including equipment specifications, manufacturing information, operational usage, transaction details, categorical attributes, and other technical characteristics.

The objective is to predict the `TargetValue` for each equipment transaction.

The competition evaluates predictions using:

> **Root Mean Squared Logarithmic Error (RMSLE)**

The project follows a complete end-to-end Machine Learning workflow:

* Data loading and validation
* Exploratory Data Analysis (EDA)
* Statistical analysis
* Missing-value analysis
* Feature engineering
* Data preprocessing
* Baseline model comparison
* Target log transformation
* Model evaluation
* Hyperparameter tuning
* Final model training
* Kaggle submission generation

---

## Dataset

The dataset consists of:

| Dataset           |    Rows | Columns |
| ----------------- | ------: | ------: |
| Training          | 138,701 |      50 |
| Test              |  15,000 |      49 |
| Metadata          |      50 |       2 |
| Sample Submission |  15,000 |       2 |

### Files

```text
train.csv
test.csv
sample_submission.csv
metadata.csv
```

### Target

The target variable is:

```text
TargetValue
```

The test dataset contains `TransactionID`, which is used to associate predictions with the corresponding transactions.

---

## Evaluation Metric

The competition uses **Root Mean Squared Logarithmic Error (RMSLE)**.

RMSLE is particularly useful for price prediction problems where relative differences between predicted and actual values are important and where the target can span a wide range.

The metric is:

```text
RMSLE = sqrt(1/n * Σ(log(1 + predicted) - log(1 + actual))²)
```

Lower RMSLE indicates better performance.

During model development, RMSLE was used as the primary metric, while **MAE, RMSE, and R²** were also calculated to provide additional insight into model performance.

---

# Exploratory Data Analysis

The initial analysis included:

### Dataset Validation

* Checking the relationship between training and test columns
* Checking duplicate records
* Checking unique `TransactionID` values
* Inspecting data types
* Examining the target distribution
* Measuring missing values
* Measuring categorical feature cardinality

### Target Analysis

The distribution of `TargetValue` was examined using descriptive statistics and visualization.

### Numerical Features

The following features were specifically investigated:

* `ManufactureYear`
* `OperationalHoursMeter`

Their distributions and relationships with the target were analyzed using histograms, boxplots, and scatterplots.

### Categorical Features

Important categorical variables were analyzed against the target, including:

* `UtilizationTier`
* `CabinType`
* `AssetScaleFactor`

### Transaction Analysis

`TransactionDate` was converted into useful temporal information, including:

* Transaction year
* Transaction month

---

# Data Cleaning & Feature Engineering

Several domain-oriented features were engineered to provide additional information to the models.

## Invalid Manufacture Years

The dataset contained a special value of `1001` for missing/invalid manufacture years.

There were:

```text
14,719
```

such records, representing approximately:

```text
10.61%
```

of the training data.

Instead of treating `1001` as a genuine manufacture year:

1. A `ManufactureYearMissing` indicator was created.
2. `1001` was replaced with `NaN`.
3. The resulting missing value was handled during preprocessing.

---

## Date Features

`TransactionDate` was decomposed into:

```text
TransactionYear
TransactionMonth
TransactionQuarter
TransactionWeekday
```

The original `TransactionDate` was subsequently excluded from the model features.

---

## Machine Age

Machine age was derived as:

```text
MachineAge = TransactionYear - ManufactureYear
```

This provides a more direct representation of equipment age than the raw manufacture year.

---

## Usage Rate

A usage-related feature was created:

```text
HoursPerYear = OperationalHoursMeter / (MachineAge + 1)
```

This provides an approximate measure of equipment usage relative to its age.

---

## Manufacture Decade

The manufacture year was also converted into:

```text
ManufactureDecade
```

This allows the models to capture broader manufacturing-era effects.

---

## Machine Age Group

Machines were grouped into age categories:

```text
0-5
6-10
11-20
21-30
31-50
50+
```

---

# Preprocessing

The preprocessing pipeline was designed separately for numerical and categorical features.

### Numerical Features

Missing numerical values were handled using **median imputation**.

### Categorical Features

Categorical missing values were handled using **most-frequent imputation** and then encoded using:

```text
OrdinalEncoder
```

Unknown categories encountered during validation or prediction were assigned an encoded value of `-1`.

The preprocessing pipeline was fitted only on the training data and then reused for validation/test data to avoid preprocessing leakage.

After feature engineering:

```text
Total features: 56
Numerical features: 12
Categorical features: 44
```

The training/validation split used:

```text
80% Training
20% Validation
random_state = 42
```

---

# Models

Four classical Machine Learning regression models were evaluated:

1. Random Forest
2. LightGBM
3. CatBoost
4. XGBoost

---

# Baseline Model Comparison

The initial models were trained using the original target values.

| Model             |     RMSLE ↓ |    MAE ↓ |       RMSE ↓ |       R² ↑ |
| :---------------- | ----------: | -------: | -----------: | ---------: |
| **Random Forest** | **0.23115** | 5,984.13 |     8,896.65 |     0.8845 |
| CatBoost          |     0.23547 | 6,247.37 |     8,856.91 |     0.8855 |
| XGBoost           |     0.23624 | 6,120.76 | **8,798.79** | **0.8870** |
| LightGBM          |     0.26117 | 6,762.55 |     9,531.32 |     0.8674 |

The initial Random Forest model achieved the lowest RMSLE among the baseline models.

However, the boosting models showed potential for further improvement through target transformation and hyperparameter optimization.

---

# Log-Transformed Target

Because the competition metric is RMSLE, the target was transformed using:

```python
y_log = np.log1p(y)
```

Models were trained on the transformed target and predictions were converted back using:

```python
prediction = np.expm1(prediction_log)
```

This approach was tested with Random Forest, CatBoost, XGBoost, and LightGBM.

### Log-Transformed Model Results

| Model               |     RMSLE ↓ |        MAE ↓ |       RMSE ↓ |       R² ↑ |
| :------------------ | ----------: | -----------: | -----------: | ---------: |
| Random Forest (Log) |     0.22582 |     5,954.39 |     9,103.94 |     0.8791 |
| CatBoost (Log)      |     0.22102 |     6,108.30 |     8,919.15 |     0.8839 |
| **XGBoost (Log)**   | **0.21392** | **5,762.26** | **8,580.13** | **0.8926** |
| LightGBM (Log)      |     0.21962 |     5,968.33 |     8,825.32 |     0.8863 |

The log-transformed XGBoost model substantially improved upon its original baseline RMSLE.

---

# Hyperparameter Tuning

Since XGBoost and LightGBM performed strongly after log transformation, hyperparameter tuning was performed using `RandomizedSearchCV`.

A custom RMSLE scorer was created so that the hyperparameter search could directly optimize the competition metric.

```python
def rmsle_score(y_true, y_pred):
    y_pred = np.maximum(y_pred, 0)
    y_true = np.maximum(y_true, 0)
    return root_mean_squared_log_error(y_true, y_pred)
```

The scorer was configured so that lower RMSLE was preferred.

---

## XGBoost Tuning

The search explored parameters including:

* `n_estimators`
* `max_depth`
* `learning_rate`
* `subsample`
* `colsample_bytree`
* `min_child_weight`
* `gamma`

The tuning used:

```text
RandomizedSearchCV
5 parameter combinations
3-fold cross-validation
```

---

## LightGBM Tuning

The LightGBM search explored:

* `n_estimators`
* `learning_rate`
* `num_leaves`
* `max_depth`
* `min_child_samples`
* `feature_fraction`
* `bagging_fraction`

The best configuration found was:

```text
n_estimators      = 7500
learning_rate     = 0.025
num_leaves        = 96
max_depth         = 10
min_child_samples = 40
feature_fraction  = 0.75
bagging_fraction  = 0.75
```

---

# Final Model Comparison

After hyperparameter tuning:

| Model              | Validation RMSLE ↓ | Validation MAE ↓ | Validation RMSE ↓ | Validation R² ↑ |
| :----------------- | -----------------: | ---------------: | ----------------: | --------------: |
| XGBoost (HPT)      |            0.21257 |         5,706.52 |          8,522.83 |          0.8940 |
| **LightGBM (HPT)** |        **0.20461** |     **5,387.54** |      **8,047.98** |      **0.9055** |

The tuned LightGBM model achieved the best validation performance across the evaluated models.

### Final validation performance

```text
RMSLE : 0.20461
MAE   : 5387.54
RMSE  : 8047.98
R²    : 0.9055
```

---

# Final Model

The final model was a LightGBM regressor trained on the log-transformed target.

```text
LightGBM
├── n_estimators      = 7500
├── learning_rate     = 0.025
├── num_leaves        = 96
├── max_depth         = 10
├── min_child_samples = 40
├── feature_fraction  = 0.75
├── bagging_fraction  = 0.75
└── random_state      = 42
```

The model was trained using the complete training dataset after the model-selection process.

Predictions were generated for the test set and converted back from log space using:

```python
np.expm1()
```

The resulting predictions were inserted into the competition submission format:

```text
TransactionID,TargetValue
```

---

# Project Workflow

```text
Raw Dataset
     │
     ▼
Data Loading
     │
     ▼
Data Validation
     │
     ▼
Exploratory Data Analysis
     │
     ▼
Data Cleaning
     │
     ▼
Feature Engineering
     │
     ├── Date Features
     ├── Machine Age
     ├── Usage Rate
     ├── Manufacture Decade
     └── Machine Age Group
     │
     ▼
Train / Validation Split
     │
     ▼
Preprocessing Pipeline
     │
     ├── Numerical Imputation
     └── Categorical Imputation + Encoding
     │
     ▼
Baseline Models
     │
     ├── Random Forest
     ├── LightGBM
     ├── CatBoost
     └── XGBoost
     │
     ▼
Log Target Transformation
     │
     ▼
Model Comparison
     │
     ▼
Hyperparameter Tuning
     │
     ├── XGBoost
     └── LightGBM
     │
     ▼
Best Model: LightGBM
     │
     ▼
Train on Full Dataset
     │
     ▼
Test Prediction
     │
     ▼
Submission.csv
```

---

# Technologies Used

### Programming

* Python

### Data Analysis

* NumPy
* Pandas
* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* XGBoost
* LightGBM
* CatBoost

### Environment

* Kaggle Notebook
* GPU acceleration for CatBoost/LightGBM/XGBoost experiments

---

# Key Takeaways

* The dataset contains a mixture of numerical, categorical, temporal, and technical features.
* Data validation and missing-value analysis were important because some fields contained invalid or missing representations.
* Feature engineering around machine age, usage, manufacturing era, and transaction timing provided additional information for the models.
* Log-transforming the target improved the performance of the boosting models under RMSLE.
* XGBoost and LightGBM benefited significantly from hyperparameter tuning.
* The tuned LightGBM model produced the best validation performance with an RMSLE of **0.20461** and an R² of **0.9055**.

---

# Repository Structure

```text
Heavy-Equipment-Selling-Price-Prediction/
│
├── Heavy_Equipment_Selling_Price_Prediction.ipynb
├── README.md
└── submission.csv
```

> Dataset files are provided by the competition and are not included in this repository.

---

# Author

**Soumyadip Hazari**

BS Degree in Data Science and Applications
Indian Institute of Technology Madras

GitHub: [SoumyadipHazari](https://github.com/SoumyadipHazari)

---

## Competition

**Heavy Equipment Selling Price Prediction Challenge**

MLP Project 2026 T2

Evaluation Metric: **Root Mean Squared Logarithmic Error (RMSLE)**
