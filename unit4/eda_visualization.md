```markdown
# Unit-4: Exploratory Data Analysis (EDA) and Visualizations

Perfect! Let me explain **Exploratory Data Analysis (EDA)** from the very beginning. Think of EDA as being a detective investigating your data before solving the case!

---

## What is EDA and Why Do We Need It?

### The Detective Analogy

Imagine you're a detective at a crime scene. Would you:
- **Immediately accuse someone?** ❌ (This is like building a model without understanding data)
- **First investigate the scene, gather clues, understand patterns?** ✅ (This is EDA!)

**EDA is the investigation phase** where you:
1. Understand what your data looks like
2. Find patterns and relationships
3. Spot problems (missing values, outliers)
4. Decide what to do next

---

## 1. DESCRIPTIVE STATISTICS

These are **summary numbers** that describe your data in simple terms.

### The Basics: Understanding a Dataset

Let's say you have test scores: `[65, 70, 75, 80, 85, 90, 95]`

### A. MEAN (Average)

**What it is:** Add all numbers and divide by count.

**Formula:** (65 + 70 + 75 + 80 + 85 + 90 + 95) / 7 = 80

```python
import numpy as np

scores = [65, 70, 75, 80, 85, 90, 95]
mean = np.mean(scores)
print(f"Mean (Average): {mean}")  # 80
```

**What it tells you:** The "center" of your data.

**Problem:** Sensitive to extreme values!
- Scores: [10, 70, 75, 80, 85, 90, 95]
- Mean = 72 (pulled down by the 10)

---

### B. MEDIAN (Middle Value)

**What it is:** The middle number when sorted.

```python
scores = [65, 70, 75, 80, 85, 90, 95]
# Sorted: [65, 70, 75, 80, 85, 90, 95]
#                      ↑ (middle)
median = np.median(scores)
print(f"Median: {median}")  # 80
```

**Why better than mean sometimes?**
```python
# With outlier
scores_with_outlier = [10, 70, 75, 80, 85, 90, 95]
print(f"Mean: {np.mean(scores_with_outlier)}")     # 72 (affected)
print(f"Median: {np.median(scores_with_outlier)}") # 80 (not affected!)
```

**Use median for:** Salaries, house prices (data with outliers)

---

### C. MODE (Most Common)

**What it is:** The value that appears most often.

```python
from scipy import stats

scores = [70, 75, 80, 80, 80, 85, 90]
mode = stats.mode(scores, keepdims=True)
print(f"Mode: {mode.mode[0]}")  # 80 (appears 3 times)
```

**Use for:** Categorical data (most popular color, most common age group)

---

### D. VARIANCE (How Spread Out?)

**What it is:** How far numbers are from the mean, on average.

**Think of it like:** Two classes both have average score of 70:
- Class A: [69, 70, 71] → Very consistent (low variance)
- Class B: [20, 70, 120] → All over the place (high variance)

```python
class_a = [69, 70, 71]
class_b = [20, 70, 120]

print(f"Class A variance: {np.var(class_a)}")  # 0.67 (consistent)
print(f"Class B variance: {np.var(class_b)}")  # 1666.67 (chaotic)
```

**Formula (don't worry too much):** Average of squared differences from mean

---

### E. STANDARD DEVIATION (Square Root of Variance)

**What it is:** More intuitive version of variance (same units as original data).

```python
print(f"Class A std: {np.std(class_a)}")  # 0.82
print(f"Class B std: {np.std(class_b)}")  # 40.82
```

**Rule of thumb (68-95-99.7 rule):**
- 68% of data within 1 standard deviation of mean
- 95% within 2 standard deviations
- 99.7% within 3 standard deviations

---

### F. SKEWNESS (Is Data Balanced?)

**What it is:** Measures if data leans to one side.

```python
from scipy.stats import skew

# Normal distribution (balanced)
normal = [2, 3, 4, 5, 6, 7, 8]
print(f"Skewness: {skew(normal)}")  # ~0 (symmetric)

# Right skewed (tail on right)
income = [30000, 35000, 40000, 45000, 50000, 500000]  # CEO salary!
print(f"Skewness: {skew(income)}")  # Positive (right skewed)

# Left skewed (tail on left)
age_at_death = [65, 70, 75, 80, 85, 90, 95, 98, 25]  # One young person
print(f"Skewness: {skew(age_at_death)}")  # Negative (left skewed)
```

**Visual understanding:**

```
Symmetric (skew ≈ 0):     Right skewed (positive):   Left skewed (negative):
      *                          *                              *
    * * *                      * * *                          * * *
  * * * * *                  * * * * * →                  ← * * * * *
```

**When to care:** If skewness > 1 or < -1, consider log transformation!

---

### G. KURTOSIS (How Pointy?)

**What it is:** Measures how "peaked" or "flat" your data is.

```python
from scipy.stats import kurtosis

# High kurtosis (sharp peak, few outliers)
peaked = [50, 50, 50, 50, 51, 51, 51, 49, 49, 49]
print(f"Kurtosis: {kurtosis(peaked)}")  # Negative (flat)

# Low kurtosis (flat, many outliers)
flat = [10, 30, 45, 50, 55, 70, 90]
print(f"Kurtosis: {kurtosis(flat)}")  # More negative (flatter)
```

**Interpretation:**
- **High kurtosis:** Sharp peak, heavy tails (many outliers)
- **Low kurtosis:** Flat distribution

**Honestly:** Kurtosis is less commonly used in practice. Focus on mean, median, std dev, and skewness first!

---

### Complete Example: Analyzing a Dataset

```python
import pandas as pd
import numpy as np

# Sample data: Student grades
data = {
    'student': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
    'math': [85, 90, 78, 92, 88, 76, 95, 89],
    'science': [80, 85, 90, 88, 92, 75, 94, 86]
}

df = pd.DataFrame(data)

# Get all statistics at once!
print(df.describe())
```

**Output:**
```
           math    science
count      8.00       8.00
mean      86.63      86.25
std        6.48       6.23
min       76.00      75.00
25%       82.75      83.75
50%       88.50      87.00    ← This is median
75%       90.75      90.50
max       95.00      94.00
```

**What this tells you:**
- Average math score: 86.63
- Middle score (median): 88.50
- Most students score between 82.75 and 90.75 (25th to 75th percentile)

---

## 2. DATA VISUALIZATION

**Why visualize?** Because humans understand pictures better than numbers!

### Setting Up

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Make plots look nice
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
```

---

### A. LINE PLOTS (For Trends Over Time)

**Use when:** You want to see how something changes over time.

```python
# Example: Temperature over days
days = [1, 2, 3, 4, 5, 6, 7]
temperature = [72, 75, 73, 78, 80, 79, 82]

plt.plot(days, temperature, marker='o', linewidth=2, markersize=8)
plt.title('Temperature Over Week', fontsize=16)
plt.xlabel('Day', fontsize=12)
plt.ylabel('Temperature (°F)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()
```

**Real use cases:**
- Stock prices over time
- Website traffic by day
- Sales trends

---

### B. BAR PLOTS (For Comparing Categories)

**Use when:** Comparing different groups or categories.

```python
# Example: Sales by product
products = ['Laptop', 'Phone', 'Tablet', 'Watch']
sales = [45000, 67000, 23000, 15000]

plt.bar(products, sales, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
plt.title('Sales by Product', fontsize=16)
plt.xlabel('Product', fontsize=12)
plt.ylabel('Sales ($)', fontsize=12)
plt.xticks(rotation=45)

# Add value labels on bars
for i, v in enumerate(sales):
    plt.text(i, v + 1000, f'${v:,}', ha='center', fontsize=10)

plt.tight_layout()
plt.show()
```

**When to use:**
- Comparing categories (sales, ratings, counts)
- Survey results
- Grade distribution

---

### C. HISTOGRAMS (Distribution of Data)

**Use when:** You want to see how data is distributed.

```python
# Example: Test scores distribution
np.random.seed(42)
scores = np.random.normal(75, 10, 1000)  # 1000 students, mean=75, std=10

plt.hist(scores, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Distribution of Test Scores', fontsize=16)
plt.xlabel('Score', fontsize=12)
plt.ylabel('Number of Students', fontsize=12)
plt.axvline(scores.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {scores.mean():.1f}')
plt.legend()
plt.show()
```

**What you learn:**
- Is data normal (bell-shaped)?
- Are there outliers?
- Is it skewed?

---

### D. SCATTER PLOTS (Relationships Between Two Variables)

**Use when:** Checking if two things are related.

```python
# Example: Study hours vs exam scores
study_hours = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
exam_scores = [45, 55, 60, 65, 70, 75, 80, 85, 88, 92]

plt.scatter(study_hours, exam_scores, s=100, alpha=0.6, color='purple')
plt.title('Study Hours vs Exam Scores', fontsize=16)
plt.xlabel('Hours Studied', fontsize=12)
plt.ylabel('Exam Score', fontsize=12)

# Add trend line
z = np.polyfit(study_hours, exam_scores, 1)
p = np.poly1d(z)
plt.plot(study_hours, p(study_hours), "r--", alpha=0.8, label='Trend')
plt.legend()
plt.show()
```

**What you see:**
- **Positive correlation:** More study → Higher scores
- **Negative correlation:** More absences → Lower scores
- **No correlation:** Shoe size → Test scores (random)

---

### E. BOX PLOTS (See Outliers and Spread)

**Use when:** Comparing distributions and finding outliers.

```python
# Example: Salaries by department
data = {
    'Engineering': [80000, 85000, 90000, 92000, 95000, 200000],  # One outlier!
    'Marketing': [60000, 65000, 68000, 70000, 72000, 75000],
    'Sales': [55000, 58000, 62000, 65000, 70000, 75000]
}

df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))

plt.figure(figsize=(10, 6))
df.boxplot()
plt.title('Salary Distribution by Department', fontsize=16)
plt.ylabel('Salary ($)', fontsize=12)
plt.xticks(rotation=45)
plt.show()
```

**Reading a box plot:**
```
    |     ← Max (excluding outliers)
    |
  ┌─┴─┐
  │   │   ← 75th percentile (Q3)
  ├───┤   ← Median (50th percentile)
  │   │   ← 25th percentile (Q1)
  └─┬─┘
    |
    |     ← Min (excluding outliers)
    
    •     ← Outlier (dot outside whiskers)
```

---

### F. PAIR PLOTS (Multiple Variables at Once)

**Use when:** Exploring relationships between ALL pairs of variables.

```python
# Example: Iris dataset (famous dataset)
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Create pairplot
sns.pairplot(df, hue='species', palette='Set2', diag_kind='hist')
plt.suptitle('Iris Dataset Pairplot', y=1.02, fontsize=16)
plt.show()
```

**What you get:**
- Diagonal: Distribution of each variable
- Off-diagonal: Scatter plots between pairs
- Color-coded by category

**Instantly see:**
- Which variables are correlated?
- Which features separate classes well?
- Are there clusters?

---

### G. HEATMAPS (Correlation Matrix)

**Use when:** Seeing which variables move together.

```python
# Example: Correlations in housing data
df = pd.DataFrame({
    'price': [200, 250, 300, 350, 400],
    'sqft': [1000, 1200, 1500, 1800, 2000],
    'bedrooms': [2, 3, 3, 4, 4],
    'age': [10, 15, 5, 20, 8]
})

# Calculate correlations
corr_matrix = df.corr()

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, fmt='.2f')
plt.title('Correlation Heatmap', fontsize=16)
plt.show()
```

**Reading the heatmap:**
- **1.0:** Perfect positive correlation (move together)
- **-1.0:** Perfect negative correlation (move opposite)
- **0:** No correlation

---

## 3. PANDAS PROFILING (Automated EDA)

**The lazy way:** Get a complete report automatically!

```python
# Install first: pip install ydata-profiling

from ydata_profiling import ProfileReport
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')

# Generate automated report
profile = ProfileReport(df, title="Data Analysis Report", explorative=True)

# Save as HTML
profile.to_file("my_report.html")
```

**What you get automatically:**
- Overview statistics
- Missing values analysis
- Correlations
- Distributions of all variables
- Warnings about issues
- Sample data

**Perfect for:** Quick initial exploration of new datasets!

---

## 4. CORRELATION ANALYSIS

**Question:** Do two variables move together?

### A. PEARSON CORRELATION (For Linear Relationships)

**Measures:** Straight-line relationships

```python
import numpy as np
from scipy.stats import pearsonr

# Example: Hours studied vs Score
hours = [1, 2, 3, 4, 5, 6, 7, 8]
scores = [50, 55, 65, 70, 75, 80, 85, 90]

correlation, p_value = pearsonr(hours, scores)
print(f"Pearson Correlation: {correlation:.3f}")  # ~0.99 (very strong!)
print(f"P-value: {p_value:.3f}")  # <0.05 means significant
```

**Interpretation:**
- **1.0:** Perfect positive (↗)
- **0.7 to 1.0:** Strong positive
- **0.3 to 0.7:** Moderate positive
- **0 to 0.3:** Weak
- **-0.3 to 0:** Weak negative
- **-1.0:** Perfect negative (↘)

---

### B. SPEARMAN CORRELATION (For Non-Linear Relationships)

**Use when:** Relationship isn't a straight line, but one increases as other increases.

```python
from scipy.stats import spearmanr

# Example: Ranking correlation
# Student rank in class vs SAT rank
class_rank = [1, 2, 3, 4, 5]
sat_rank = [2, 1, 3, 5, 4]

correlation, p_value = spearmanr(class_rank, sat_rank)
print(f"Spearman Correlation: {correlation:.3f}")
```

**When to use Spearman over Pearson:**
- Data is ranked/ordinal
- Relationship is monotonic but not linear
- Data has outliers

---

## 5. DETECTING TRENDS AND ANOMALIES

### A. MOVING AVERAGES (Smooth Out Noise)

**Use when:** Data is noisy, you want to see the trend.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Example: Daily sales with noise
dates = pd.date_range('2024-01-01', periods=100)
np.random.seed(42)
sales = 100 + np.cumsum(np.random.randn(100)) + np.random.randn(100) * 10

df = pd.DataFrame({'date': dates, 'sales': sales})

# Calculate moving average
df['MA_7'] = df['sales'].rolling(window=7).mean()  # 7-day average
df['MA_30'] = df['sales'].rolling(window=30).mean()  # 30-day average

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['sales'], alpha=0.3, label='Daily Sales')
plt.plot(df['date'], df['MA_7'], linewidth=2, label='7-Day MA')
plt.plot(df['date'], df['MA_30'], linewidth=2, label='30-Day MA')
plt.title('Sales with Moving Averages', fontsize=16)
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**What moving averages do:**
- Smooth out daily fluctuations
- Show underlying trend
- Help identify turning points

---

### B. DETECTING ANOMALIES (Finding Weird Data)

**Method 1: Z-Score (Statistical)**

```python
from scipy import stats

# Example: Website response times
response_times = [100, 105, 98, 102, 107, 500, 103, 99]  # 500ms is weird!

# Calculate z-scores
z_scores = np.abs(stats.zscore(response_times))

# Flag outliers (z-score > 3)
outliers = np.where(z_scores > 3)[0]
print(f"Outlier indices: {outliers}")  # [5] (the 500ms value)
print(f"Outlier values: {[response_times[i] for i in outliers]}")
```

**Method 2: IQR (Interquartile Range)**

```python
# Calculate quartiles
Q1 = np.percentile(response_times, 25)
Q3 = np.percentile(response_times, 75)
IQR = Q3 - Q1

# Define outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Find outliers
outliers = [x for x in response_times if x < lower_bound or x > upper_bound]
print(f"Outliers: {outliers}")  # [500]
```

**Method 3: Visual (Box Plot)**

```python
plt.boxplot(response_times)
plt.title('Response Times - Outliers Visible')
plt.ylabel('Time (ms)')
plt.show()
```

---

## COMPLETE EDA WORKFLOW EXAMPLE

Let me show you a full analysis from start to finish:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load data (example: house prices)
df = pd.DataFrame({
    'price': [200000, 250000, 300000, 350000, 400000, 1500000, 220000],
    'sqft': [1000, 1200, 1500, 1800, 2000, 3500, 1100],
    'bedrooms': [2, 3, 3, 4, 4, 6, 2],
    'age': [10, 15, 5, 20, 8, 3, 12]
})

print("=" * 50)
print("STEP 1: BASIC INFO")
print("=" * 50)
print(df.info())
print("\n", df.head())

print("\n" + "=" * 50)
print("STEP 2: DESCRIPTIVE STATISTICS")
print("=" * 50)
print(df.describe())

print("\n" + "=" * 50)
print("STEP 3: CHECK FOR MISSING VALUES")
print("=" * 50)
print(df.isnull().sum())

print("\n" + "=" * 50)
print("STEP 4: SKEWNESS CHECK")
print("=" * 50)
for col in df.select_dtypes(include=[np.number]).columns:
    skewness = stats.skew(df[col])
    print(f"{col}: {skewness:.2f}", end="")
    if abs(skewness) > 1:
        print(" ⚠️ HIGHLY SKEWED!")
    else:
        print()

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Distribution of prices
axes[0, 0].hist(df['price'], bins=10, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Price Distribution')
axes[0, 0].set_xlabel('Price ($)')
axes[0, 0].set_ylabel('Count')

# 2. Correlation heatmap
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=axes[0, 1], center=0)
axes[0, 1].set_title('Correlation Matrix')

# 3. Scatter: sqft vs price
axes[1, 0].scatter(df['sqft'], df['price'], s=100, alpha=0.6)
axes[1, 0].set_title('Square Feet vs Price')
axes[1, 0].set_xlabel('Square Feet')
axes[1, 0].set_ylabel('Price ($)')

# 4. Box plot: detect outliers
df.boxplot(column='price', ax=axes[1, 1])
axes[1, 1].set_title('Price - Outlier Detection')

plt.tight_layout()
plt.show()

print("\n" + "=" * 50)
print("STEP 5: CORRELATION ANALYSIS")
print("=" * 50)
print(corr)

print("\n" + "=" * 50)
print("STEP 6: KEY INSIGHTS")
print("=" * 50)
print("✓ Price is highly correlated with sqft (", f"{corr.loc['price', 'sqft']:.2f})")
print("✓ One outlier detected in price ($1,500,000)")
print("✓ Price distribution is right-skewed")
print("✓ Consider log transformation for price")
```

---

## QUICK REFERENCE CHEAT SHEET

| Task | Tool | Code |
|------|------|------|
| **Summary stats** | Pandas | `df.describe()` |
| **Distribution** | Histogram | `plt.hist(data)` |
| **Relationships** | Scatter | `plt.scatter(x, y)` |
| **Categories** | Bar plot | `plt.bar(categories, values)` |
| **Trend** | Line plot | `plt.plot(x, y)` |
| **Outliers** | Box plot | `plt.boxplot(data)` |
| **Correlations** | Heatmap | `sns.heatmap(df.corr())` |
| **All pairs** | Pairplot | `sns.pairplot(df)` |
| **Automated** | Profiling | `ProfileReport(df)` |

---

## Summary

**EDA is your investigation phase before building models. It helps you:**

1. **Understand your data** - What's the shape, size, types?
2. **Find problems** - Missing values, outliers, errors
3. **Discover patterns** - Correlations, trends, distributions
4. **Make decisions** - Which features to use? What preprocessing is needed?

**Key Steps:**
1. Load data and check basic info
2. Calculate descriptive statistics
3. Visualize distributions
4. Check for correlations
5. Identify and handle outliers
6. Document findings

**Remember:** Never skip EDA! It's like a doctor examining a patient before surgery. The better you understand your data, the better your model will perform.

---

Does this help you understand EDA better? Would you like me to:
1. Work through a real dataset together?
2. Explain any specific visualization in more detail?
3. Show you how to interpret results for machine learning?
```
