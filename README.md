# Datasets
**TestA.csv:**`sha256:d679a638036f47bc6f04d8ad22090ee73df0041af0fe62c8f5dc22e85a2cb94`
**Trian.csv:**`sha256:54122ab08d4c32f71281e294e9fc41452b3ea47a934247d135a193b4bf5aae74`

# Financial Risk Control Model Replication and Loan Default Prediction

## Project Objective
This project is based on the Tianchi Financial Risk Control competition dataset, constructing machine learning models to predict whether users will default on loans. The project comprehensively reproduces and optimizes the financial risk prediction process from data exploration, feature engineering, and model training to result evaluation, aiming to provide a reproducible, interpretable, and optimizable risk control model framework.

## Data Introduction
- **Data Source**: Loan records from a credit platform (anonymized)
- **Data Scale**: 800,000 records in training set, 200,000 records in Test A, 200,000 records in Test B
- **Number of Features**: 47 features total, including 15 anonymous variables (n0-n14)
- **Key Features**:
  - Loan amount (loanAmnt)
  - Interest rate (interestRate)
  - Credit grade (grade, subGrade)
  - Employment length (employmentLength)
  - Annual income (annualIncome)
  - Delinquency history (delinquency_2years)
  - Credit score range (ficoRangeLow, ficoRangeHigh)
  - Anonymous features (n0-n14)
- **Target Variable**: `isDefault` (whether default occurred, 1 indicates default, 0 indicates no default)

## Data Processing Pipeline

### 1. Data Loading and Preliminary Exploration
- Load training and test sets using pandas
- Basic data statistics: sample count, feature dimensions, missing values
- Data type analysis: 42 numerical features, 5 object-type features

### 2. Feature Classification
Further classification of numerical features:
- **Continuous Variables**: such as loanAmnt, annualIncome, interestRate, etc.
- **Discrete Variables**: such as term, homeOwnership, verificationStatus, etc.
- **Single-value Variables**: such as id (no modeling value)

### 3. Missing Value Handling
Identify and calculate missing proportions for each feature:
- Some features like employmentLength, postCode, dti have missing values
- Anonymous features n0-n14 have varying degrees of missing data

### 4. Feature Engineering
- LabelEncoder encoding for categorical features
- Standardization of numerical features (MinMaxScaler)
- Feature selection using SelectKBest
- Time feature processing (issueDate, etc.)

### 5. Data Splitting
- Training set and validation set splitting (train_test_split)
- Stratified sampling to ensure consistent class distribution (StratifiedKFold)

## Modeling Methods

### Model Selection
This project employs ensemble learning methods, combining the following three powerful gradient boosting models:

1. **XGBoost**: Extreme Gradient Boosting, efficient and accurate
2. **LightGBM**: Lightweight Gradient Boosting Framework developed by Microsoft
3. **CatBoost**: Gradient boosting algorithm particularly effective with categorical features

### Evaluation Metrics
- **AUC (Area Under ROC Curve)**: Primary evaluation metric measuring overall ranking ability
- **Accuracy**: Auxiliary evaluation metric
- **F1 Score**: Balance between precision and recall
- **Log Loss**: Measures accuracy of probability predictions

### Training Strategy
- Use cross-validation to prevent overfitting
- Ensemble multiple model results (weighted averaging or voting)
- Parameter tuning optimization: grid search or Bayesian optimization

## Model Results
According to competition requirements, the final submission is the probability of default (y=1) for each test sample. Model performance on the test set is as follows:

- **Best Single Model AUC**: Approximately 0.72-0.75
- **Ensemble Model AUC**: Approximately 0.76-0.78
- **Key Feature Importance**:
  1. Credit score-related features (ficoRangeLow/High)
  2. Loan amount and interest rate
  3. Income to debt ratio (dti)
  4. Some features from the anonymous n-series features
  5. 
**Future Optimization Directions**

-More In-depth Feature Engineering (Feature Crosses, Polynomial Features)
-Outlier Detection and Handling
-Utilizing Deep Learning Models (e.g., TabNet, DeepFM)
-Model Interpretability Analysis (SHAP, LIME)
-Online Learning and Model Update Strategies
