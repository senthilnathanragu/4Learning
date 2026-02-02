# Unit-3: Data Preprocessing Techniques
## Complete Learning Guide for Beginners

---

## Table of Contents
1. Introduction to Data Preprocessing
2. Outlier Detection (Z-score & IQR)
3. Handling Skewed Data
4. Feature Scaling
5. Encoding Categorical Data
6. Data Splitting
7. Lab Experiments with Complete Code

---

## 1. INTRODUCTION TO DATA PREPROCESSING

### What is Data Preprocessing?
Data preprocessing is the process of preparing raw data for analysis and machine learning. Real-world data is messy and contains:
- Missing values
- Outliers
- Inconsistent formats
- Categorical variables
- Data on different scales

### Why is it Important?
- **Better model performance**: Clean data = accurate predictions
- **Reduces errors**: Removes noise and anomalies
- **Improves efficiency**: Processing becomes faster
- **Standardization**: Makes all features comparable

### The Data Preprocessing Pipeline:
```
Raw Data → Detection → Cleaning → Transformation → Encoding → Scaling → Split → Ready for ML
```

---

## 2. OUTLIER DETECTION

### What are Outliers?
Outliers are extreme values that are significantly different from other observations. They can be:
- **Valid outliers**: Rare but real events
- **Invalid outliers**: Errors in data collection

### Method 1: Z-Score Method

#### Concept:
The Z-score measures how many standard deviations a data point is from the mean.

**Formula**: `Z = (X - mean) / standard_deviation`

**Rule**: If |Z| > 3, it's an outlier (99.7% of data falls within ±3 std dev)

#### How it Works:
```
Step 1: Calculate mean and standard deviation
Step 2: Calculate Z-score for each value
Step 3: Identify values where |Z| > 3
Step 4: Remove or handle outliers
```

#### Code Example:

```python
import numpy as np
import pandas as pd
from scipy import stats

# Sample data with outliers
data = np.array([10, 12, 11, 13, 12, 100, 11, 12, 13, 11])
# Notice 100 is an outlier

# Method 1: Using NumPy and Pandas
z_scores = np.abs(stats.zscore(data))
print("Z-scores:", z_scores)

# Identify outliers (Z > 3)
outliers_mask = z_scores > 3
print("Outlier positions:", np.where(outliers_mask))

# Remove outliers
data_cleaned = data[~outliers_mask]
print("Original data:", data)
print("Cleaned data:", data_cleaned)

# Method 2: Using Pandas (easier for DataFrames)
df = pd.DataFrame({'values': data})
z_scores_df = np.abs(stats.zscore(df['values']))
df_cleaned = df[z_scores_df < 3]
print("\nOriginal DataFrame:\n", df)
print("\nCleaned DataFrame:\n", df_cleaned)
```

**Output Explanation:**
- Z-scores close to 0: Normal values
- Z-scores > 3 or < -3: Outliers

---

### Method 2: IQR (Interquartile Range) Method

#### Concept:
IQR is the range between the 25th percentile (Q1) and 75th percentile (Q3).

**Formula:**
```
IQR = Q3 - Q1
Lower Bound = Q1 - 1.5 * IQR
Upper Bound = Q3 + 1.5 * IQR
Outliers: Values < Lower Bound OR > Upper Bound
```

#### How it Works:
```
Step 1: Calculate Q1 (25th percentile) and Q3 (75th percentile)
Step 2: Calculate IQR = Q3 - Q1
Step 3: Calculate bounds
Step 4: Identify values outside bounds
Step 5: Remove or handle outliers
```

#### Code Example:

```python
import pandas as pd
import numpy as np

# Sample data
data = np.array([10, 12, 11, 13, 12, 100, 11, 12, 13, 11])
df = pd.DataFrame({'values': data})

# Calculate Q1, Q3, and IQR
Q1 = df['values'].quantile(0.25)
Q3 = df['values'].quantile(0.75)
IQR = Q3 - Q1

print(f"Q1 (25th percentile): {Q1}")
print(f"Q3 (75th percentile): {Q3}")
print(f"IQR: {IQR}")

# Calculate bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Lower Bound: {lower_bound}")
print(f"Upper Bound: {upper_bound}")

# Identify outliers
outliers_mask = (df['values'] < lower_bound) | (df['values'] > upper_bound)
print(f"\nOutliers: {df[outliers_mask].values}")

# Remove outliers
df_cleaned = df[~outliers_mask]
print(f"\nOriginal length: {len(df)}")
print(f"Cleaned length: {len(df_cleaned)}")
print(f"Removed: {len(df) - len(df_cleaned)} outlier(s)")
```

#### Comparison: Z-Score vs IQR
```
| Aspect          | Z-Score    | IQR            |
|-----------------|------------|----------------|
| Uses            | Mean, StDev| Quartiles      |
| Sensitive to    | Extreme    | Less sensitive |
| Best for        | Normal     | Non-normal     |
| Threshold       | |Z| > 3    | < Q1-1.5*IQR   |
```

---

## 3. HANDLING SKEWED DATA

### What is Skewed Data?
Skewed data has an asymmetrical distribution where values cluster more on one side.

**Types:**
- **Right skew (Positive)**: Tail on the right, most values on left
- **Left skew (Negative)**: Tail on the left, most values on right

### Why Handle Skewness?
- Improves model performance
- Makes data more normal-like
- Stabilizes variance
- Reduces impact of outliers

### Method: Log Transformation

#### Concept:
Apply logarithm to reduce the range and skewness of data.

**Formula**: `X_transformed = log(X)`

**When to use:** For right-skewed data with positive values

#### Code Example:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create right-skewed data
data = np.array([1, 2, 2, 3, 3, 3, 4, 5, 8, 15, 100])

# Create DataFrame
df = pd.DataFrame({'original': data})

# Apply log transformation
df['log_transformed'] = np.log(df['original'])

print("Original data:")
print(df['original'].values)
print(f"Skewness: {df['original'].skew()}")

print("\nLog-transformed data:")
print(df['log_transformed'].values)
print(f"Skewness: {df['log_transformed'].skew()}")

# Visualize (plot if possible)
print("\nDataFrame:")
print(df)

# Note: Skewness close to 0 = symmetric, >1 or <-1 = highly skewed
```

#### Other Transformations:
```python
# Square root transformation (less aggressive than log)
df['sqrt_transformed'] = np.sqrt(df['original'])

# Box-Cox transformation (finds optimal lambda automatically)
from scipy.stats import boxcox
df['boxcox_transformed'], lambda_param = boxcox(df['original'])
print(f"Optimal lambda: {lambda_param}")
```

---

## 4. FEATURE SCALING

### What is Feature Scaling?
Scaling transforms features to a similar range so they have equal importance in ML models.

### Why Scale Features?
- Algorithms like KNN, SVM work better with scaled data
- Gradient descent converges faster
- Prevents features with larger ranges from dominating
- Improves model accuracy and training speed

### Method 1: Min-Max Normalization (Standardization to [0,1])

#### Concept:
Scales each feature to a [0, 1] range.

**Formula**: `X_scaled = (X - min) / (max - min)`

#### Code Example:

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Sample data with different ranges
data = np.array([[100, 50000],
                 [200, 60000],
                 [150, 55000],
                 [300, 75000]])

# Create DataFrame
df = pd.DataFrame(data, columns=['Age', 'Salary'])
print("Original data:")
print(df)
print(f"Age range: {df['Age'].min()} - {df['Age'].max()}")
print(f"Salary range: {df['Salary'].min()} - {df['Salary'].max()}")

# Method 1: Using sklearn (recommended)
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=['Age', 'Salary'])

print("\nMin-Max Scaled data (0-1 range):")
print(df_scaled)
print(f"Age range: {df_scaled['Age'].min()} - {df_scaled['Age'].max()}")

# Method 2: Manual scaling
df_manual = (df - df.min()) / (df.max() - df.min())
print("\nManual Min-Max scaling:")
print(df_manual)
```

**Output:** All values between 0 and 1, preserving relationships

---

### Method 2: Standardization (Z-Score Normalization)

#### Concept:
Transforms data to have mean = 0 and standard deviation = 1.

**Formula**: `X_scaled = (X - mean) / standard_deviation`

#### Code Example:

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Sample data
data = np.array([[100, 50000],
                 [200, 60000],
                 [150, 55000],
                 [300, 75000]])

df = pd.DataFrame(data, columns=['Age', 'Salary'])
print("Original data:")
print(df)
print(f"Age - Mean: {df['Age'].mean()}, StDev: {df['Age'].std()}")

# Method 1: Using sklearn (recommended)
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=['Age', 'Salary'])

print("\nStandardized data (mean=0, std=1):")
print(df_scaled)
print(f"Age - Mean: {df_scaled['Age'].mean()}, StDev: {df_scaled['Age'].std()}")

# Method 2: Manual standardization
df_manual = (df - df.mean()) / df.std()
print("\nManual standardization:")
print(df_manual)
```

**Output:** Values centered around 0 with spread of 1

#### Comparison: Min-Max vs Standardization
```
| Aspect        | Min-Max         | Standardization |
|---------------|-----------------|-----------------|
| Range         | [0, 1]          | (-∞, +∞)        |
| Preserves     | Shape           | Relationships   |
| Outlier effect| Sensitive       | Less sensitive  |
| Best for      | NN, Image data  | Linear models   |
```

---

## 5. ENCODING CATEGORICAL DATA

### What are Categorical Features?
Features with discrete values (categories), not continuous numbers.

**Examples:**
- Color: Red, Green, Blue
- Country: USA, UK, Canada
- Size: Small, Medium, Large

### Why Encode?
ML algorithms need numeric input, not text strings.

---

### Method 1: Label Encoding

#### Concept:
Converts each category to a unique integer (0, 1, 2, ...).

**Best for:** Ordinal data (where order matters)

#### Code Example:

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Sample data
df = pd.DataFrame({
    'Color': ['Red', 'Blue', 'Green', 'Red', 'Blue'],
    'Size': ['Small', 'Large', 'Medium', 'Small', 'Large']
})

print("Original data:")
print(df)

# Method 1: Using sklearn
le_color = LabelEncoder()
df['Color_encoded'] = le_color.fit_transform(df['Color'])

le_size = LabelEncoder()
df['Size_encoded'] = le_size.fit_transform(df['Size'])

print("\nAfter Label Encoding:")
print(df)

# See the mapping
print("\nColor mapping:")
print(dict(zip(le_color.classes_, le_color.transform(le_color.classes_))))

# Method 2: Using Pandas
df['Color_map'] = pd.factorize(df['Color'])[0]
print("\nUsing Pandas factorize:")
print(df[['Color', 'Color_map']])

# Reverse mapping (convert back)
print("\nReverse mapping:")
print(le_color.inverse_transform([0, 1, 2]))
```

**Output:** Red→0, Blue→1, Green→2 (arbitrary assignment)

---

### Method 2: One-Hot Encoding

#### Concept:
Creates binary columns for each category (1 if present, 0 if not).

**Best for:** Nominal data (no order relationship)

#### Code Example:

```python
import pandas as pd

# Sample data
df = pd.DataFrame({
    'Color': ['Red', 'Blue', 'Green', 'Red', 'Blue'],
    'Size': ['Small', 'Large', 'Medium', 'Small', 'Large']
})

print("Original data:")
print(df)

# Method 1: Using Pandas get_dummies (simplest)
df_encoded = pd.get_dummies(df, columns=['Color', 'Size'])
print("\nAfter One-Hot Encoding (get_dummies):")
print(df_encoded)

# Method 2: Using sklearn OneHotEncoder (more control)
from sklearn.preprocessing import OneHotEncoder
import numpy as np

ohe = OneHotEncoder(sparse_output=False)
encoded = ohe.fit_transform(df[['Color', 'Size']])
columns = ohe.get_feature_names_out(['Color', 'Size'])
df_encoded2 = pd.DataFrame(encoded, columns=columns)

print("\nUsing sklearn OneHotEncoder:")
print(df_encoded2)

# Get feature names
print("\nFeature names:")
print(list(columns))
```

**Output:**
```
Original: Color='Red'
Becomes:  Color_Blue=0, Color_Green=0, Color_Red=1
```

---

### Method 3: Ordinal Encoding

#### Concept:
Assigns integers based on order/hierarchy of categories.

**Best for:** Ordinal data like "Low" < "Medium" < "High"

#### Code Example:

```python
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# Sample ordinal data
df = pd.DataFrame({
    'Education': ['High School', 'Bachelor', 'Master', 'PhD', 'Bachelor'],
    'Rating': ['Poor', 'Good', 'Excellent', 'Poor', 'Good']
})

print("Original data:")
print(df)

# Define order
education_order = [['High School', 'Bachelor', 'Master', 'PhD']]
rating_order = [['Poor', 'Good', 'Excellent']]

# Apply Ordinal Encoding
oe = OrdinalEncoder(categories=education_order + rating_order)
df_encoded = pd.DataFrame(
    oe.fit_transform(df),
    columns=['Education_encoded', 'Rating_encoded']
)

print("\nAfter Ordinal Encoding:")
print(df_encoded)

# Manual mapping (alternative method)
edu_map = {'High School': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
rating_map = {'Poor': 1, 'Good': 2, 'Excellent': 3}

df['Education_manual'] = df['Education'].map(edu_map)
df['Rating_manual'] = df['Rating'].map(rating_map)

print("\nManual Ordinal Encoding:")
print(df[['Education', 'Education_manual', 'Rating', 'Rating_manual']])
```

**Output:** Maintains order: Poor(1) < Good(2) < Excellent(3)

#### When to Use Which:
```
| Encoding    | Use Case              | Example              |
|-------------|----------------------|----------------------|
| Label      | Ordinal categories   | Education level      |
| One-Hot    | Nominal categories   | Color, Country       |
| Ordinal    | Ordered categories   | Rating (Poor-Good)   |
```

---

## 6. DATA SPLITTING

### What is Data Splitting?
Dividing data into separate sets for training, validation, and testing.

### Why Split Data?
- **Training set**: Teaches the model
- **Testing set**: Evaluates real-world performance
- **Validation set**: Tunes hyperparameters (optional)

### Common Split Ratios:
- **2-way split**: 80% train, 20% test
- **3-way split**: 60% train, 20% validation, 20% test
- **For small datasets**: 70% train, 30% test

### Code Example:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Create sample dataset
np.random.seed(42)  # For reproducibility
data = {
    'Age': np.random.randint(20, 60, 100),
    'Salary': np.random.randint(30000, 120000, 100),
    'Experience': np.random.randint(0, 30, 100),
    'Department': np.random.choice(['Sales', 'IT', 'HR', 'Finance'], 100),
    'Promoted': np.random.choice([0, 1], 100)
}
df = pd.DataFrame(data)

print("Original dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Method 1: Basic 80-20 split
X = df.drop('Promoted', axis=1)  # Features (all except target)
y = df['Promoted']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% for testing
    random_state=42     # For reproducibility
)

print(f"\nTraining set: {X_train.shape}")
print(f"Testing set: {X_test.shape}")

# Method 2: 3-way split (60-20-20)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.25,  # 0.25 of 0.8 = 0.2 (20% of original)
    random_state=42
)

print(f"\nTraining set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Testing set: {X_test.shape}")

# Stratified split (maintains class distribution for imbalanced data)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Ensures same proportion in train and test
)

print(f"\nStratified split:")
print(f"Train - Class distribution:\n{y_train.value_counts()}")
print(f"Test - Class distribution:\n{y_test.value_counts()}")
```

---

## 7. COMPLETE PREPROCESSING PIPELINE

### Putting It All Together:

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import stats

# Step 1: Load data
df = pd.read_csv('data.csv')

# Step 2: Handle missing values
df = df.dropna()  # or df.fillna(df.mean())

# Step 3: Detect and remove outliers (Z-score method)
z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
df = df[(z_scores < 3).all(axis=1)]

# Step 4: Handle skewed numerical data
df['income_log'] = np.log(df['income'] + 1)  # +1 to avoid log(0)

# Step 5: Encode categorical variables
le = LabelEncoder()
df['department_encoded'] = le.fit_transform(df['department'])

# Step 6: Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Step 7: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 8: Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Pipeline complete! Ready for ML models")
```

---

## LAB EXPERIMENTS

### EXPERIMENT 1: Outlier Detection and Removal (Z-Score Method)

**Objective**: Detect and remove outliers from a dataset using the Z-score method.

**Dataset**: Student marks (100 marks scale, with some data entry errors)

#### Complete Code:

```python
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Step 1: Create a dataset with outliers
np.random.seed(42)
student_marks = {
    'Student_ID': range(1, 51),
    'Math': np.concatenate([
        np.random.normal(75, 10, 48),  # Normal marks
        np.array([1, 999])  # Two outliers (data entry errors)
    ]),
    'English': np.concatenate([
        np.random.normal(70, 12, 49),
        np.array([500])  # One outlier
    ]),
    'Science': np.concatenate([
        np.random.normal(80, 8, 50)  # No outliers
    ])
}

df = pd.DataFrame(student_marks)

print("="*60)
print("EXPERIMENT 1: OUTLIER DETECTION (Z-SCORE METHOD)")
print("="*60)

print("\n[STEP 1] Original Dataset:")
print(f"Shape: {df.shape}")
print(f"First 5 rows:")
print(df.head())
print(f"\nLast 5 rows (containing outliers):")
print(df.tail())

print(f"\n[STEP 2] Dataset Statistics:")
print(df.describe())

# Step 2: Calculate Z-scores for all numerical columns
print("\n[STEP 3] Z-Score Calculation:")
z_scores = np.abs(stats.zscore(df[['Math', 'English', 'Science']]))
z_scores_df = pd.DataFrame(z_scores, columns=['Math_Z', 'English_Z', 'Science_Z'])

print("Z-scores for first 5 students:")
print(z_scores_df.head())
print("\nZ-scores for last 5 students (with outliers):")
print(z_scores_df.tail())

# Step 3: Identify outliers (|Z| > 3)
print("\n[STEP 4] Identifying Outliers (|Z| > 3):")
outlier_mask = (z_scores > 3).any(axis=1)
outlier_indices = np.where(outlier_mask)[0]

print(f"Number of outliers found: {len(outlier_indices)}")
if len(outlier_indices) > 0:
    print(f"Outlier Student IDs: {df.iloc[outlier_indices]['Student_ID'].values}")
    print("\nOutlier Details:")
    print(df.iloc[outlier_indices])
    print("\nTheir Z-scores:")
    print(z_scores_df.iloc[outlier_indices])

# Step 4: Remove outliers
df_cleaned = df[~outlier_mask].reset_index(drop=True)

print("\n[STEP 5] Dataset After Removing Outliers:")
print(f"Original shape: {df.shape}")
print(f"Cleaned shape: {df_cleaned.shape}")
print(f"Rows removed: {df.shape[0] - df_cleaned.shape[0]}")

print(f"\nCleaned dataset statistics:")
print(df_cleaned.describe())

print("\n[STEP 6] Comparison:")
print("\nOriginal Math marks - Mean:", df['Math'].mean(), ", Std:", df['Math'].std())
print("Cleaned Math marks - Mean:", df_cleaned['Math'].mean(), ", Std:", df_cleaned['Math'].std())

print("\nOriginal English marks - Mean:", df['English'].mean(), ", Std:", df['English'].std())
print("Cleaned English marks - Mean:", df_cleaned['English'].mean(), ", Std:", df_cleaned['English'].std())

print("\n" + "="*60)
print("EXPERIMENT 1 COMPLETE!")
print("="*60)

# Optional: Also show IQR method for comparison
print("\n\n[BONUS] SAME DATA USING IQR METHOD:")
print("="*60)

def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outlier_mask = (data[column] < lower_bound) | (data[column] > upper_bound)
    return outlier_mask, lower_bound, upper_bound

print("\nIQR Method Results:")
for col in ['Math', 'English', 'Science']:
    mask, lower, upper = detect_outliers_iqr(df, col)
    n_outliers = mask.sum()
    print(f"\n{col}:")
    print(f"  Lower bound: {lower:.2f}, Upper bound: {upper:.2f}")
    print(f"  Outliers found: {n_outliers}")
    if n_outliers > 0:
        print(f"  Outlier values: {df[mask][col].values}")
```

#### Expected Output:
```
Original shape: (50, 3)
Cleaned shape: (47, 3)
Rows removed: 3

Outliers found at Student IDs: [49, 50, ...]
Removed values like 999, 1, 500 that were clearly errors
```

---

### EXPERIMENT 2: Train-Test Split with Label Encoding

**Objective**: Split a dataset into training and testing subsets and apply label encoding to categorical columns.

**Dataset**: Employee information (includes categorical features)

#### Complete Code:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Step 1: Create a dataset with categorical features
np.random.seed(42)
employees = {
    'Employee_ID': range(1, 101),
    'Age': np.random.randint(25, 60, 100),
    'Department': np.random.choice(['Sales', 'IT', 'HR', 'Finance'], 100),
    'Position': np.random.choice(['Junior', 'Senior', 'Manager'], 100),
    'Salary': np.random.randint(30000, 120000, 100),
    'Experience_Years': np.random.randint(0, 35, 100),
    'Performance_Rating': np.random.choice(['Low', 'Medium', 'High'], 100),
    'Promoted': np.random.choice([0, 1], 100)
}

df = pd.DataFrame(employees)

print("="*70)
print("EXPERIMENT 2: DATA SPLITTING WITH LABEL ENCODING")
print("="*70)

print("\n[STEP 1] Original Dataset:")
print(f"Total samples: {len(df)}")
print(f"Features: {list(df.columns)}")
print(f"\nFirst 10 rows:")
print(df.head(10))

print(f"\nData types:")
print(df.dtypes)

print(f"\nCategorical columns:")
categorical_cols = df.select_dtypes(include=['object']).columns
print(list(categorical_cols))

print(f"\nCategorical data samples:")
print(df[categorical_cols].head(10))

# Step 2: Separate features and target
print("\n[STEP 2] Separating Features and Target Variable:")
target = 'Promoted'
X = df.drop([target, 'Employee_ID'], axis=1)  # Remove ID and target
y = df[target]

print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print(f"Features: {list(X.columns)}")
print(f"\nTarget variable distribution:")
print(f"  0 (Not Promoted): {(y==0).sum()}")
print(f"  1 (Promoted): {(y==1).sum()}")

# Step 3: Encode categorical features BEFORE splitting
print("\n[STEP 3] Encoding Categorical Features:")
print("(Encoding before split is correct practice)")

# Create a copy to maintain original data
X_encoded = X.copy()

# Dictionary to store encoders (for later decoding if needed)
label_encoders = {}

# Encode each categorical column
for col in categorical_cols:
    if col in X_encoded.columns:  # Check if column exists in X
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col])
        label_encoders[col] = le
        
        print(f"\n{col} Encoding:")
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print(f"  Mapping: {mapping}")

print(f"\nEncoded features:")
print(X_encoded.head(10))

# Step 4: Split data (80-20 split)
print("\n[STEP 4] Splitting Data (80% Train, 20% Test):")

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Maintain class distribution
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")
print(f"Total: {X_train.shape[0] + X_test.shape[0]} samples")

print(f"\nTraining set target distribution:")
print(f"  0 (Not Promoted): {(y_train==0).sum()}")
print(f"  1 (Promoted): {(y_train==1).sum()}")

print(f"\nTesting set target distribution:")
print(f"  0 (Not Promoted): {(y_test==0).sum()}")
print(f"  1 (Promoted): {(y_test==1).sum()}")

# Step 5: Display sample data
print("\n[STEP 5] Sample Data from Each Set:")

print("\nFirst 5 rows of Training Features:")
print(X_train.head())

print("\nFirst 5 corresponding Training Targets:")
print(y_train.head().values)

print("\nFirst 5 rows of Testing Features:")
print(X_test.head())

print("\nFirst 5 corresponding Testing Targets:")
print(y_test.head().values)

# Step 6: Verify encoding worked correctly
print("\n[STEP 6] Verification:")
print(f"Training set - Any missing values: {X_train.isnull().sum().sum()}")
print(f"Testing set - Any missing values: {X_test.isnull().sum().sum()}")

print(f"\nTraining set - Data types after encoding:")
print(X_train.dtypes)

print(f"\nNo categorical columns in encoded data: {not any(X_train.dtypes == 'object')}")

# Step 7: Optional - Decode back to original values
print("\n[STEP 7] Decoding Sample (Convert Back to Original):")
sample_row = X_test.iloc[0].copy()
print(f"\nEncoded sample: {sample_row.values}")

# Decode
sample_decoded = sample_row.copy()
for col in label_encoders:
    if col in sample_decoded.index:
        sample_decoded[col] = label_encoders[col].inverse_transform([int(sample_decoded[col])])[0]

print(f"Decoded sample: {sample_decoded.values}")

# Display comparison
print("\nComparison (Original vs Encoded):")
original_sample = df.iloc[X_test.index[0]]
print(f"Department - Original: {original_sample['Department']}, Encoded: {X_test.iloc[0]['Department']}")
print(f"Position - Original: {original_sample['Position']}, Encoded: {X_test.iloc[0]['Position']}")
print(f"Performance_Rating - Original: {original_sample['Performance_Rating']}, Encoded: {X_test.iloc[0]['Performance_Rating']}")

print("\n" + "="*70)
print("EXPERIMENT 2 COMPLETE!")
print("="*70)

# Optional: Save processed data
print("\n[BONUS] Saving Processed Data:")
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

print(f"Training data shape: {train_data.shape}")
print(f"Testing data shape: {test_data.shape}")
print("(These can be saved using: train_data.to_csv('train.csv', index=False))")
```

#### Expected Output:
```
Original dataset: 100 samples
Training set: 80 samples
Testing set: 20 samples

Categorical columns encoded:
  Department: Sales→0, IT→1, HR→2, Finance→3
  Position: Junior→0, Senior→1, Manager→2
  Performance_Rating: Low→0, Medium→1, High→2

All data ready for ML models!
```

---

## KEY CONCEPTS SUMMARY

### Z-Score Outlier Detection:
- Identifies points > 3 standard deviations from mean
- Formula: Z = (X - mean) / std_dev
- Remove rows where |Z| > 3

### IQR Outlier Detection:
- Uses quartiles instead of mean/std
- More robust to extreme outliers
- Formula: Outliers outside [Q1-1.5*IQR, Q3+1.5*IQR]

### Skewed Data:
- Use log transformation for right-skewed positive data
- Reduces impact of outliers
- Makes distribution more normal-like

### Feature Scaling:
- **Min-Max**: Scales to [0,1] range
- **Standardization**: Scales to mean=0, std=1
- Essential for distance-based algorithms

### Categorical Encoding:
- **Label**: Integer assignment (ordinal data)
- **One-Hot**: Binary columns (nominal data)
- **Ordinal**: Ordered integers (ranked categories)

### Data Splitting:
- 80-20 or 70-30 for binary split
- 60-20-20 for three-way split
- Stratify for imbalanced datasets

---

## Common Mistakes to Avoid

1. **Encoding before splitting**: Always encode AFTER splitting to avoid data leakage
2. **Fitting scaler on test data**: Fit scaler on training data only, then transform test data
3. **Removing all outliers**: Some outliers are valid; check if they're errors first
4. **Forgetting to reset index**: Use `reset_index(drop=True)` after filtering
5. **Not stratifying split**: Can lead to different class distributions in train/test

---

## Resources for Further Learning

- Pandas Documentation: https://pandas.pydata.org/docs/
- Scikit-learn Preprocessing: https://scikit-learn.org/stable/modules/preprocessing.html
- NumPy Reference: https://numpy.org/doc/stable/

---

**Good luck with your teaching! This guide covers all the essential concepts with practical examples.**
