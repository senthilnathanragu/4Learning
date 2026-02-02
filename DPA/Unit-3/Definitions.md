# Data Preprocessing and Analytics Using Python
## Unit Definitions

---

## 1. Introduction to Data Preprocessing

**Data Preprocessing** is the process of transforming raw data into a clean, organized, and suitable format for machine learning algorithms. It involves handling missing values, removing noise, normalizing data, and converting data types to ensure that the dataset is accurate, complete, and ready for analysis or modeling.

**Purpose**: To improve data quality, reduce bias, enhance model performance, and ensure that machine learning algorithms can interpret the data correctly.

---

## 2. Outlier Detection

### What are Outliers?
**Outliers** are data points that significantly deviate from the majority of observations in a dataset. They can arise from measurement errors, data entry mistakes, or represent genuine extreme values that differ from the normal pattern.

**Impact**: Outliers can skew statistical analyses, distort mean values, and negatively affect the performance of machine learning models.

### Z-Score Method

**Z-Score (Standard Score)** is a statistical measurement that describes a data point's relationship to the mean of a group of values, measured in terms of standard deviations from the mean.

**Formula**: Z = (X - μ) / σ
- X = individual data point
- μ = mean of the dataset
- σ = standard deviation

**Outlier Threshold**: Typically, data points with a Z-score greater than 3 or less than -3 are considered outliers (i.e., values more than 3 standard deviations away from the mean).

**Best Used When**: The data follows a normal (Gaussian) distribution.

### IQR Method (Interquartile Range)

**IQR (Interquartile Range)** is the range between the first quartile (Q1, 25th percentile) and the third quartile (Q3, 75th percentile) of a dataset. It measures the spread of the middle 50% of the data.

**Formula**: IQR = Q3 - Q1

**Outlier Detection Rule**:
- Lower Bound = Q1 - 1.5 × IQR
- Upper Bound = Q3 + 1.5 × IQR
- Any data point below the lower bound or above the upper bound is considered an outlier.

**Best Used When**: The data is skewed or doesn't follow a normal distribution; more robust to extreme values than Z-score.

---

## 3. Handling Skewed Data

### What is Skewed Data?

**Skewness** refers to the asymmetry in the distribution of data. A dataset is skewed when the data points are not evenly distributed around the mean.

**Types of Skewness**:
- **Positive (Right) Skew**: The tail on the right side is longer; most values are concentrated on the left with few extreme high values.
- **Negative (Left) Skew**: The tail on the left side is longer; most values are concentrated on the right with few extreme low values.

### Log Transformation

**Log Transformation** is a mathematical technique that applies the logarithm function to data values to reduce skewness and make the distribution more symmetric and closer to normal.

**Common Transformations**:
- **Natural Log**: ln(x) or log(x) base e
- **Log Base 10**: log10(x)
- **Log(1+x)**: Used when data contains zeros

**Purpose**: 
- Reduces the impact of extreme values
- Compresses the range of large values
- Makes multiplicative relationships additive
- Improves model performance for algorithms that assume normality

**When to Use**: Best for positively skewed data with values greater than 0.

---

## 4. Feature Scaling

**Feature Scaling** is the process of normalizing the range of independent variables or features in a dataset. It ensures that all features contribute equally to the model and prevents features with larger magnitudes from dominating the learning process.

### Min-Max Normalization

**Min-Max Normalization** (also called Min-Max Scaling) rescales features to a fixed range, typically [0, 1], by transforming each value based on the minimum and maximum values in the feature.

**Formula**: X_scaled = (X - X_min) / (X_max - X_min)

**Characteristics**:
- Preserves the original distribution shape
- Sensitive to outliers (outliers can compress the range of normal values)
- Useful when you need bounded values in a specific range
- Commonly used in neural networks and image processing

**Range**: Typically scales values to [0, 1], but can be adjusted to any range [a, b]

### Standardization (Z-Score Normalization)

**Standardization** transforms features to have a mean of 0 and a standard deviation of 1, creating a standard normal distribution.

**Formula**: X_standardized = (X - μ) / σ
- μ = mean of the feature
- σ = standard deviation of the feature

**Characteristics**:
- Does not bound values to a specific range
- Less sensitive to outliers than min-max scaling
- Assumes features are normally distributed (or can benefit from such transformation)
- Commonly used in algorithms that assume normally distributed data (e.g., linear regression, logistic regression, SVM)

**Result**: Values typically range around -3 to +3 for normally distributed data.

---

## 5. Encoding Categorical Data

**Categorical Data** consists of discrete values that represent categories or groups rather than numerical measurements. Machine learning algorithms require numerical input, so categorical variables must be encoded into numerical format.

### One-Hot Encoding

**One-Hot Encoding** converts categorical variables into a binary matrix where each category becomes a separate binary column. For each observation, the column corresponding to its category contains a 1, while all other columns contain 0s.

**Example**:
- Original: ['Red', 'Blue', 'Green']
- Encoded: Red=[1,0,0], Blue=[0,1,0], Green=[0,0,1]

**Characteristics**:
- Creates n new columns for n categories
- No ordinal relationship is implied between categories
- Can lead to high dimensionality with many categories ("curse of dimensionality")
- Suitable for nominal (unordered) categorical variables

**When to Use**: For categorical variables with no inherent order (e.g., colors, countries, product types).

### Label Encoding

**Label Encoding** assigns a unique integer to each category in a categorical variable, converting categories into numerical labels.

**Example**:
- Original: ['Low', 'Medium', 'High']
- Encoded: Low=0, Medium=1, High=2

**Characteristics**:
- Creates a single column with integer values
- Implies an ordinal relationship (which may not exist)
- More memory-efficient than one-hot encoding
- Can mislead algorithms into assuming numerical relationships

**When to Use**: 
- For the target variable in classification problems
- For ordinal variables where order matters
- With tree-based algorithms that can handle the numerical interpretation correctly

**Caution**: Not recommended for nominal variables in linear models, as it introduces artificial ordering.

### Ordinal Encoding

**Ordinal Encoding** assigns integers to categorical variables based on their natural order or ranking. It's similar to label encoding but explicitly preserves the meaningful order of categories.

**Example**:
- Original: ['Low', 'Medium', 'High']
- Encoded: Low=1, Medium=2, High=3 (maintaining the order)

**Characteristics**:
- Preserves the inherent order in the data
- Creates a single numerical column
- The numerical differences reflect the ordinal relationships
- More interpretable than arbitrary label encoding

**When to Use**: For ordinal categorical variables with a clear ranking or hierarchy (e.g., education level: High School < Bachelor's < Master's < PhD; ratings: Poor < Fair < Good < Excellent).

---

## 6. Data Splitting

**Data Splitting** is the process of dividing a dataset into separate subsets for training, validating, and testing machine learning models. This ensures unbiased evaluation and helps detect overfitting.

### Train-Test Split

**Train-Test Split** divides the dataset into two parts:
- **Training Set**: Used to train the model (typically 70-80% of data)
- **Test Set**: Used to evaluate the final model performance on unseen data (typically 20-30% of data)

**Purpose**: To assess how well the model generalizes to new, unseen data.

### Train-Validation-Test Split

**Train-Validation-Test Split** divides the dataset into three parts:
- **Training Set**: Used to train the model (typically 60-70% of data)
- **Validation Set**: Used to tune hyperparameters and make model selection decisions (typically 15-20% of data)
- **Test Set**: Used for final, unbiased evaluation of the model (typically 15-20% of data)

**Purpose**: 
- Training set: Learns patterns from data
- Validation set: Guides model improvement and prevents overfitting during development
- Test set: Provides final, unbiased performance metrics

### sklearn.model_selection

**sklearn.model_selection** is a module in scikit-learn that provides tools for splitting datasets and performing cross-validation.

**Key Functions**:

1. **train_test_split()**: Splits data into training and testing sets
   - Parameters: test_size, random_state, shuffle, stratify
   - Returns: X_train, X_test, y_train, y_test

2. **Cross-validation methods**: Provide more robust evaluation
   - **KFold**: Splits data into k consecutive folds
   - **StratifiedKFold**: Preserves class distribution in each fold
   - **cross_val_score()**: Evaluates model using cross-validation

**Important Parameters**:
- **test_size**: Proportion of dataset for test set (e.g., 0.2 for 20%)
- **random_state**: Seed for reproducibility
- **stratify**: Ensures proportional representation of target classes in splits (important for imbalanced datasets)
- **shuffle**: Whether to shuffle data before splitting (default: True)

**Best Practices**:
- Always set random_state for reproducibility
- Use stratified splitting for classification tasks with imbalanced classes
- Keep test set completely separate until final evaluation
- Never train on test data or use test data for any model development decisions

---

## Summary

Data preprocessing is essential for building effective machine learning models. The key steps include:

1. **Detecting and handling outliers** to prevent skewed results
2. **Transforming skewed data** to improve normality
3. **Scaling features** to ensure equal contribution to the model
4. **Encoding categorical variables** to make them usable by algorithms
5. **Splitting data** properly to enable unbiased model evaluation

Each technique serves a specific purpose and should be applied thoughtfully based on the characteristics of your data and the requirements of your chosen algorithms.
