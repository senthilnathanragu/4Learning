# Unit 5: Basic Data Analytics using Python — Complete Guide

Let me teach you everything step by step, starting from scratch!

---

## 🧠 PART 1: Introduction to Machine Learning

### What is Machine Learning?

Machine Learning (ML) is teaching a computer to **learn from data** and make decisions **without being explicitly programmed**.

Think of it like this:
- **Traditional Programming:** You give rules → computer gives output
- **Machine Learning:** You give data + output → computer finds the rules

---

### Supervised vs Unsupervised Learning

**Supervised Learning** — The data has **labels** (answers are already known)
- Example: You show 1000 house photos labeled "expensive" or "cheap" → model learns to predict new ones
- Algorithms: Linear Regression, KNN, Decision Trees

**Unsupervised Learning** — The data has **no labels** (find hidden patterns)
- Example: Group customers by buying behavior without knowing categories in advance
- Algorithms: K-Means Clustering, PCA

| Feature | Supervised | Unsupervised |
|---|---|---|
| Labels | Yes | No |
| Goal | Predict | Discover patterns |
| Example | Spam detection | Customer segmentation |

---

## 📈 PART 2: Linear Regression

### What is it?

Linear Regression finds the **best straight line** through your data points to predict a continuous value.

**Formula:**  `y = mx + c`
- `y` = what you want to predict (house price)
- `x` = input feature (square footage)
- `m` = slope (how much y changes per unit of x)
- `c` = intercept (y value when x = 0)

### Visual Intuition

```
Price
 |          * *
 |       *  /
 |    * *  / ← best fit line
 |  *     /
 |       /
 +-------------- Area
```

The model finds this line automatically!

---

### 🔬 Experiment 1: Linear Regression — House Price Prediction

```python
# ============================================================
# EXPERIMENT 1: Linear Regression - Predict House Prices
# ============================================================

# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------------------------------
# Step 2: Create Sample Dataset
# (Square footage vs House Price)
# -------------------------------------------------------
data = {
    'SquareFootage': [500, 750, 1000, 1200, 1500, 1800, 2000,
                      2200, 2500, 3000, 3200, 3500, 4000, 4500, 5000],
    'Price':         [50000, 75000, 100000, 115000, 140000, 165000, 185000,
                      210000, 235000, 280000, 300000, 330000, 370000, 420000, 470000]
}

df = pd.DataFrame(data)
print("Dataset:")
print(df)
print("\nShape:", df.shape)
print("\nBasic Statistics:")
print(df.describe())

# -------------------------------------------------------
# Step 3: Prepare Data (X = input, y = output)
# -------------------------------------------------------
X = df[['SquareFootage']]  # 2D array (required by sklearn)
y = df['Price']             # 1D array

print("\nX (Features):\n", X.head())
print("\ny (Target):\n", y.head())

# -------------------------------------------------------
# Step 4: Split data into Training and Testing sets
# -------------------------------------------------------
# 80% data for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# -------------------------------------------------------
# Step 5: Create and Train the Model
# -------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)  # This is where learning happens!

print("\n--- Model Parameters ---")
print(f"Slope (m):     {model.coef_[0]:.2f}")
print(f"Intercept (c): {model.intercept_:.2f}")
print(f"Formula: Price = {model.coef_[0]:.2f} × SquareFootage + {model.intercept_:.2f}")

# -------------------------------------------------------
# Step 6: Make Predictions
# -------------------------------------------------------
y_pred = model.predict(X_test)

print("\n--- Predictions vs Actual ---")
results = pd.DataFrame({'Actual Price': y_test.values, 'Predicted Price': y_pred.astype(int)})
print(results)

# -------------------------------------------------------
# Step 7: Evaluate the Model
# -------------------------------------------------------
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"MSE  (Mean Squared Error):  {mse:.2f}")
print(f"RMSE (Root MSE):            {rmse:.2f}")
print(f"R²   (Accuracy Score):      {r2:.4f}  ({r2*100:.2f}%)")

# What is R²?
# R² = 1.0 means perfect prediction
# R² = 0.9 means model explains 90% of the variation in data

# -------------------------------------------------------
# Step 8: Predict a New House
# -------------------------------------------------------
new_house = pd.DataFrame({'SquareFootage': [2800]})
predicted_price = model.predict(new_house)
print(f"\nPredicted price for 2800 sqft house: ${predicted_price[0]:,.2f}")

# -------------------------------------------------------
# Step 9: Visualize Everything
# -------------------------------------------------------
plt.figure(figsize=(14, 5))

# Plot 1: Data + Regression Line
plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue', label='Actual Data', alpha=0.7, s=80)
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Best Fit Line')
plt.xlabel('Square Footage')
plt.ylabel('Price ($)')
plt.title('House Price vs Square Footage')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Actual vs Predicted
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, color='green', s=100, alpha=0.8)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('linear_regression_result.png', dpi=100)
plt.show()
print("\nPlot saved!")
```

---

## 🌸 PART 3: K-Nearest Neighbors (KNN)

### What is KNN?

KNN is a **classification algorithm** that classifies a new point by looking at the **K nearest neighbors** around it.

**Simple Rule:** "Tell me your neighbors and I'll tell you who you are!"

### Visual Intuition

```
  😊 😊              😊 = Class A
       😊
          ❓  ← new point
       😡 😡
  😡 😡              😡 = Class B

K=3: Look at 3 nearest neighbors
     2 are 😡, 1 is 😊 → Predict 😡
```

### How K affects the result
- **Small K (e.g., K=1):** Very sensitive, can overfit
- **Large K:** More stable but may underfit
- **Best K:** Found by testing different values

---

### The Iris Dataset

The most famous ML dataset! Contains measurements of 3 flower species:
- **Setosa**
- **Versicolor**
- **Virginica**

Features: Sepal length, Sepal width, Petal length, Petal width

---

### 🔬 Experiment 2: KNN on Iris Dataset

```python
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
knn = KNeighborsClassifier(n_neighbors=3)

# Train model
knn.fit(X_train, y_train)

# Predict
y_pred = knn.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

# Plot confusion matrix
sns.heatmap(cm, annot=True, cmap="Blues")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()```

---

## 🔵 PART 4: K-Means Clustering (Unsupervised)

```python
# ============================================================
# K-MEANS CLUSTERING — Unsupervised Learning
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

# Step 1: Create sample data (no labels!)
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)

# Step 2: Find best number of clusters using Elbow Method
inertias = []
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)
    print(f"K={k}: Inertia = {km.inertia_:.2f}")

# Step 3: Train with best K
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

print(f"\nCluster Centers:\n{centers.round(2)}")
print(f"Silhouette Score: {silhouette_score(X, labels):.3f}")
# Silhouette score: close to 1 = good clusters

# Step 4: Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Elbow plot
axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Inertia (Within-cluster sum of squares)')
axes[0].set_title('Elbow Method — Find Best K')
axes[0].axvline(x=4, color='red', linestyle='--', label='Best K=4')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Cluster plot
colors = ['red', 'blue', 'green', 'purple']
for i in range(4):
    mask = labels == i
    axes[1].scatter(X[mask, 0], X[mask, 1], c=colors[i],
                    label=f'Cluster {i+1}', alpha=0.6, s=40)
axes[1].scatter(centers[:, 0], centers[:, 1],
                c='black', marker='*', s=300, label='Centroids', zorder=5)
axes[1].set_title('K-Means Clustering Result')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('kmeans_result.png', dpi=100)
plt.show()
```

---

**Model Evaluation Techniques** are used to **measure how well a machine learning model performs**.
After training a model, we need to check whether it **predicts correctly on new (unseen) data**.

If we do not evaluate the model properly, it may **memorize the training data but fail in real-world predictions**. This problem is called **overfitting**.

So, evaluation techniques help us answer:

* Is the model accurate?
* Does it generalize well to new data?
* Which model is better?

---

# 1. Train-Test Split

This is the **simplest evaluation technique**.

The dataset is divided into **two parts**:

1. **Training Set** – Used to train the model
2. **Testing Set** – Used to evaluate the model

Example:

If we have **150 samples (Iris dataset)**

* Training data → 120 samples (80%)
* Testing data → 30 samples (20%)

Process:

1. Train model using training data
2. Test the model using test data
3. Compare predicted values with actual values

Python example:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

Here:

* `X_train` → training features
* `X_test` → testing features
* `y_train` → training labels
* `y_test` → testing labels

---

# 2. Accuracy

Accuracy is the **most common evaluation metric**.

It measures the **percentage of correct predictions**.

Formula:

Accuracy =
(Number of Correct Predictions / Total Predictions) × 100

Example:

Suppose the model tested **30 samples**.

* Correct predictions = 27
* Wrong predictions = 3

Accuracy =

27 / 30 = **0.9 (90%)**

Python example:

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
```

Limitations:

Accuracy works well when **classes are balanced**.
But if data is **imbalanced**, accuracy can be misleading.

---

# 3. Confusion Matrix

A **Confusion Matrix** shows how many predictions were correct and incorrect.

For binary classification it looks like this:

|                 | Predicted Positive  | Predicted Negative  |
| --------------- | ------------------- | ------------------- |
| Actual Positive | True Positive (TP)  | False Negative (FN) |
| Actual Negative | False Positive (FP) | True Negative (TN)  |

Meaning:

**True Positive (TP)**
Model correctly predicts positive.

**True Negative (TN)**
Model correctly predicts negative.

**False Positive (FP)**
Model predicts positive but actually negative.

**False Negative (FN)**
Model predicts negative but actually positive.

Example:

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)
```

Example output:

```
[[10 0 0]
 [0 8 2]
 [0 1 9]]
```

This shows how many predictions were correct or incorrect for each class.

---

# 4. Precision

Precision measures **how many predicted positives are actually correct**.

Formula:

Precision =
TP / (TP + FP)

Example:

Model predicted **10 emails as spam**

* 8 are actually spam
* 2 are not spam

Precision =

8 / (8 + 2) = **0.8**

Meaning:

80% of predicted spam emails were actually spam.

Python example:

```python
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred, average='macro')
```

---

# 5. Recall (Sensitivity)

Recall measures **how many actual positives were correctly predicted**.

Formula:

Recall =
TP / (TP + FN)

Example:

Suppose there are **10 spam emails**

Model detects **8**

Recall =

8 / 10 = **0.8**

Meaning:

Model detected **80% of spam emails**.

Python example:

```python
from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred, average='macro')
```

---

# 6. F1 Score

F1 Score is the **balance between Precision and Recall**.

Formula:

F1 Score =

2 × (Precision × Recall) / (Precision + Recall)

It is useful when:

* Data is **imbalanced**
* Both **precision and recall are important**

Python example:

```python
from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred, average='macro')
```

---

# 7. Cross Validation

Cross Validation is a **better evaluation technique than train-test split**.

Instead of splitting data once, it splits data **multiple times**.

Most common method:

**K-Fold Cross Validation**

Example: **5-fold cross validation**

Steps:

1. Dataset divided into **5 parts**
2. Train on **4 parts**
3. Test on **1 part**
4. Repeat **5 times**
5. Average the results

Benefits:

* More reliable evaluation
* Uses the entire dataset

Python example:

```python
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()

scores = cross_val_score(model, X, y, cv=5)

print(scores)
print(scores.mean())
```

---

# Summary of Model Evaluation Techniques

Train-Test Split
→ Splits dataset into training and testing

Accuracy
→ Percentage of correct predictions

Confusion Matrix
→ Detailed prediction results

Precision
→ Correct positive predictions

Recall
→ Correct detection of actual positives

F1 Score
→ Balance between precision and recall


For **Regression models**, we cannot use accuracy like classification.
Instead, we measure **how far the predicted value is from the actual value**.

The most common **Model Evaluation Techniques for Regression** are:

* **MAE (Mean Absolute Error)**
* **MSE (Mean Squared Error)**
* **R² Score (Coefficient of Determination)**

---

# 1. MAE – Mean Absolute Error

MAE = \frac{1}{n}\sum_{i=1}^{n} |y_i - \hat{y}_i|

### Meaning

MAE calculates the **average absolute difference between actual values and predicted values**.

* (y_i) → actual value
* (\hat{y}_i) → predicted value
* (n) → number of samples

### Key idea

It measures **how wrong the predictions are on average**.

Absolute value means **negative errors become positive**.

---

### Example

Actual house prices:

```
[200, 250, 300]
```

Predicted prices:

```
[210, 240, 290]
```

Errors:

```
|200 - 210| = 10
|250 - 240| = 10
|300 - 290| = 10
```

MAE:

```
(10 + 10 + 10) / 3 = 10
```

Meaning:

**On average, the model prediction is off by 10 units.**

---

### Advantages

* Easy to understand
* Not sensitive to outliers

---

### Python Example

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
print("MAE:", mae)
```

---

# 2. MSE – Mean Squared Error

MSE = \frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2

### Meaning

MSE calculates the **average of squared errors**.

Instead of absolute values, the errors are **squared**.

---

### Example

Actual values:

```
[200, 250, 300]
```

Predicted values:

```
[210, 240, 290]
```

Errors:

```
200 - 210 = -10
250 - 240 = 10
300 - 290 = 10
```

Squared errors:

```
(-10)^2 = 100
10^2 = 100
10^2 = 100
```

MSE:

```
(100 + 100 + 100) / 3 = 100
```

---

### Key idea

Because errors are **squared**, large errors get **penalized more heavily**.

Example:

```
Error = 2  → 4
Error = 10 → 100
```

So MSE strongly penalizes **large mistakes**.

---

### Advantages

* Good for optimization
* Used in most regression algorithms

---

### Python Example

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)
```

---

# 3. R² Score (Coefficient of Determination)

R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}

### Meaning

R² measures **how well the model explains the variance in the data**.

It tells **how good the model is compared to a simple average prediction**.

---

### Value Range

```
R² = 1      → Perfect model
R² = 0      → Model is same as predicting mean
R² < 0      → Model is worse than mean prediction
```

---

### Example

Suppose:

```
R² = 0.85
```

This means:

**The model explains 85% of the variation in the data.**

---

### Interpretation

| R² Score | Meaning            |
| -------- | ------------------ |
| 1        | Perfect prediction |
| 0.9      | Very good model    |
| 0.7      | Good model         |
| 0.5      | Moderate model     |
| 0        | Poor model         |

---

### Python Example

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)
```

---

# Simple Comparison

| Metric   | Meaning                | Sensitive to Outliers |
| -------- | ---------------------- | --------------------- |
| MAE      | Average absolute error | No                    |
| MSE      | Average squared error  | Yes                   |
| R² Score | Model goodness         | No                    |

---

# Quick Summary

**MAE**

Average absolute difference between predicted and actual values.

**MSE**

Average squared difference between predicted and actual values.

**R² Score**

Shows **how well the model explains the data**.

## 📊 PART 5: Model Evaluation Techniques

```python
# ============================================================
# MODEL EVALUATION — All Metrics Explained
# ============================================================

from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score,
                              mean_squared_error, r2_score,
                              confusion_matrix)
import numpy as np

print("=" * 50)
print("CLASSIFICATION METRICS")
print("=" * 50)

# Sample actual vs predicted values
y_actual    = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
y_predicted = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]

cm = confusion_matrix(y_actual, y_predicted)
TP = cm[1][1]  # True Positive
TN = cm[0][0]  # True Negative
FP = cm[0][1]  # False Positive (said yes, actually no)
FN = cm[1][0]  # False Negative (said no, actually yes)

print(f"\nConfusion Matrix:")
print(f"      Predicted 0  Predicted 1")
print(f"Actual 0    {TN}           {FP}     ← TN={TN}, FP={FP}")
print(f"Actual 1    {FN}           {TP}     ← FN={FN}, TP={TP}")

accuracy  = accuracy_score(y_actual, y_predicted)
precision = precision_score(y_actual, y_predicted)
recall    = recall_score(y_actual, y_predicted)
f1        = f1_score(y_actual, y_predicted)

print(f"\nAccuracy  = (TP+TN)/(Total)  = {accuracy:.2f}  → {accuracy*100:.0f}% correct")
print(f"Precision = TP/(TP+FP)       = {precision:.2f}  → Of predicted positives, how many are real?")
print(f"Recall    = TP/(TP+FN)       = {recall:.2f}  → Of actual positives, how many did we catch?")
print(f"F1-Score  = 2×(P×R)/(P+R)   = {f1:.2f}  → Balance of Precision & Recall")

print("\n" + "=" * 50)
print("REGRESSION METRICS")
print("=" * 50)

y_actual_reg    = [100, 200, 300, 400, 500]
y_predicted_reg = [110, 190, 310, 390, 490]

mse  = mean_squared_error(y_actual_reg, y_predicted_reg)
rmse = np.sqrt(mse)
r2   = r2_score(y_actual_reg, y_predicted_reg)
mae  = np.mean(np.abs(np.array(y_actual_reg) - np.array(y_predicted_reg)))

print(f"\nMAE  (Mean Absolute Error): {mae:.2f}  → Average error in original units")
print(f"MSE  (Mean Squared Error):  {mse:.2f}   → Penalizes large errors more")
print(f"RMSE (Root MSE):            {rmse:.2f}  → Same units as target variable")
print(f"R²   (R-Squared):           {r2:.4f} → {r2*100:.0f}% variance explained (1.0 = perfect)")
```

---

## 🌍 PART 6: Real-World Use Cases

### Use Case 1: Spam Detection

```python
# ============================================================
# USE CASE: Email Spam Detection using KNN
# ============================================================

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Features: [word_count, exclamation_marks, links, caps_ratio]
# Label: 1=Spam, 0=Not Spam
np.random.seed(42)

spam_emails = np.random.rand(100, 4) * [500, 20, 15, 1]
spam_emails[:, 1] += 10  # Spam has more exclamation marks
spam_emails[:, 2] += 5   # Spam has more links
spam_emails[:, 3] = np.random.uniform(0.5, 1.0, 100)  # High caps ratio

ham_emails = np.random.rand(100, 4) * [300, 5, 3, 0.3]

X = np.vstack([spam_emails, ham_emails])
y = np.array([1]*100 + [0]*100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)

print("=== SPAM DETECTION RESULTS ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=['Not Spam', 'Spam']))

# Test a new email
new_email = np.array([[450, 18, 12, 0.8]])  # Suspicious email
new_email_s = scaler.transform(new_email)
result = model.predict(new_email_s)
print(f"\nNew Email → {'🚨 SPAM!' if result[0]==1 else '✅ Not Spam'}")
```

---

## 📋 Quick Reference Summary

| Concept | Type | Goal | Algorithm | Metric |
|---|---|---|---|---|
| House Price Prediction | Supervised / Regression | Predict a number | Linear Regression | R², RMSE |
| Flower Classification | Supervised / Classification | Predict a category | KNN | Accuracy, F1 |
| Customer Grouping | Unsupervised / Clustering | Find groups | K-Means | Silhouette Score |
| Spam Detection | Supervised / Classification | Yes or No | KNN / Naive Bayes | Precision, Recall |

---

## 🛠️ Setup — Install Required Libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## 🎯 Key Takeaways for Teaching

Here's how you can explain each concept to students simply:

**Linear Regression** → "Drawing the best line through data points to predict future values"

**KNN** → "You're judged by the company you keep — look at your nearest neighbors"

**K-Means** → "Automatically group similar things together without labels"

**Confusion Matrix** → "A report card of where the model got confused"

**R² Score** → "How much of the story does our model explain?" (1.0 = 100%)

**Train/Test Split** → "Study with some questions, test with different questions — just like real exams!"

---

Start by running Experiment 1 first (simpler), then Experiment 2. Each code block has step-by-step comments so you can walk students through every line. Let me know if you want me to go deeper on any specific concept!
