# 2-Day Teaching Plan for Unit-4: EDA and Visualizations

## **DAY 1: Foundations of EDA and Basic Visualizations**

### **Session 1 (Morning - 2 hours)**

#### **1. Introduction to EDA (30 minutes)**
- What is EDA and why it matters in the data pipeline
- The big picture: EDA's role in data preprocessing and analytics
- Types of EDA: Univariate vs Multivariate
- Real-world examples of EDA insights

#### **2. Descriptive Statistics (90 minutes)**

**A. Measures of Central Tendency (30 minutes)**
- Mean, Median, Mode
- When to use which measure
- Practical examples with Python
- Hands-on: Calculate for sample datasets

**B. Measures of Spread (30 minutes)**
- Variance and Standard Deviation
- Range and IQR
- Understanding data dispersion
- Hands-on: Compute spread metrics

**C. Measures of Shape (30 minutes)**
- Skewness (left, right, symmetric)
- Kurtosis (outlier presence)
- Interpreting these metrics
- Hands-on: Analyze distributions

**Break (15 minutes)**

### **Session 2 (Late Morning/Early Afternoon - 2 hours)**

#### **3. Data Visualization Basics (2 hours)**

**A. Setting Up & Line Plots (30 minutes)**
- Introduction to Matplotlib and Seaborn
- Basic plotting syntax
- Line plots for trends over time
- Hands-on: Create time series plots

**B. Bar Plots (30 minutes)**
- Vertical and horizontal bar charts
- Grouped and stacked bars
- When to use bar plots
- Hands-on: Compare categories

**C. Histograms (30 minutes)**
- Understanding distributions
- Choosing bin sizes
- Density plots
- Hands-on: Visualize data distributions

**D. Box Plots (30 minutes)**
- Components of a box plot
- Identifying outliers visually
- Comparing multiple groups
- Hands-on: Create comparative box plots

**Lunch Break (1 hour)**

### **Session 3 (Afternoon - 2 hours)**

#### **4. Scatter Plots and Relationships (1 hour)**
- Basic scatter plots
- Adding regression lines
- Color-coding for 3rd dimension
- Interpreting relationships
- Hands-on: Explore variable relationships

#### **5. Lab Experiment 1 - Part A (1 hour)**
**Practical Session: Scatter Plots and Basic Relationships**
- Students work on provided dataset
- Create multiple scatter plots
- Identify positive/negative correlations
- Document observations
- Q&A and troubleshooting

### **Day 1 Wrap-up (15 minutes)**
- Recap of key concepts
- Preview of Day 2
- Assignment: Practice creating different plot types with a given dataset

---

## **DAY 2: Advanced EDA Techniques and Automation**

### **Session 1 (Morning - 2 hours)**

#### **1. Quick Recap of Day 1 (15 minutes)**
- Review descriptive statistics
- Review basic visualizations
- Address questions from Day 1

#### **2. Pairplots for Multivariate Analysis (45 minutes)**
- What are pairplots and when to use them
- Understanding the pairplot matrix
- Diagonal vs off-diagonal elements
- Color-coding by categories
- Hands-on: Create and interpret pairplots

**Break (15 minutes)**

#### **3. Correlation Analysis (45 minutes)**

**A. Pearson Correlation (20 minutes)**
- Understanding linear relationships
- Correlation coefficient interpretation
- Assumptions and limitations
- Hands-on: Calculate Pearson correlations

**B. Spearman Correlation (15 minutes)**
- Monotonic vs linear relationships
- When to use Spearman over Pearson
- Handling ordinal data
- Hands-on: Compare Pearson vs Spearman

**C. Correlation Heatmaps (10 minutes)**
- Visualizing correlation matrices
- Identifying multicollinearity
- Hands-on: Create correlation heatmaps

### **Session 2 (Late Morning - 1.5 hours)**

#### **4. Detecting Trends and Anomalies (1.5 hours)**

**A. Moving Averages (45 minutes)**
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Smoothing noisy data
- Identifying trends
- Hands-on: Apply moving averages to time series

**B. Anomaly Detection (45 minutes)**
- Z-score method
- IQR method
- Visual identification with box plots
- Hands-on: Detect outliers in datasets

**Lunch Break (1 hour)**

### **Session 3 (Afternoon - 2.5 hours)**

#### **5. Pandas Profiling for Automated EDA (45 minutes)**
- Introduction to ydata-profiling
- Installation and setup
- Generating automated reports
- Interpreting report sections:
  - Overview
  - Variables
  - Correlations
  - Missing values
  - Sample data
- Hands-on: Generate a profile report

**Break (15 minutes)**

#### **6. Lab Experiment 1 - Part B (30 minutes)**
**Complete Lab 1: Pairplots and Relationships**
- Work with multivariate dataset
- Create comprehensive pairplots
- Analyze all variable relationships
- Document findings

#### **7. Lab Experiment 2 (1 hour)**
**Complete Lab 2: Pandas Profiling and Data Quality**
- Load dataset with quality issues
- Generate pandas profiling report
- Identify missing data patterns
- Find correlations automatically
- Detect outliers and anomalies
- Create data quality summary
- Propose cleaning strategies

### **Session 4 (Final Session - 1 hour)**

#### **8. Complete EDA Workflow Integration (30 minutes)**
- Step-by-step EDA process
- Combining all techniques learned
- From raw data to insights
- Best practices and common pitfalls
- Real-world case study walkthrough

#### **9. Review and Q&A (30 minutes)**
- Recap of all concepts
- Address student questions
- Discuss practical applications
- Career relevance of EDA skills
- Resources for further learning

---

## **Quick Reference: Concept Distribution**

### **DAY 1 CONCEPTS ✓**
✅ Introduction to EDA  
✅ Descriptive Statistics (Mean, Median, Mode)  
✅ Variance and Standard Deviation  
✅ Skewness and Kurtosis  
✅ Line Plots  
✅ Bar Plots  
✅ Histograms  
✅ Box Plots  
✅ Scatter Plots (basic)  
✅ Lab Experiment 1 - Part A (Scatter plots)  

### **DAY 2 CONCEPTS ✓**
✅ Pairplots  
✅ Pearson Correlation  
✅ Spearman Correlation  
✅ Correlation Heatmaps  
✅ Moving Averages  
✅ Anomaly Detection (Z-score, IQR)  
✅ Pandas Profiling  
✅ Lab Experiment 1 - Part B (Pairplots)  
✅ Lab Experiment 2 (Complete - Pandas Profiling)  
✅ Integrated EDA Workflow  

---

## **Teaching Tips for Success**

### **For Day 1:**
- Start with WHY EDA matters - motivate students
- Use visual examples early to grab attention
- Make descriptive statistics relatable (exam scores, salaries)
- Keep code snippets short and runnable
- Allow students to experiment with parameters
- End with something visually impressive (scatter plots)

### **For Day 2:**
- Begin with a quick win (review + pairplots)
- Show the "wow factor" of automated profiling
- Emphasize time-saving aspects
- Focus on interpretation, not just creation
- Labs should be guided but allow creativity
- Connect everything back to real projects
- End on a high note - show career relevance

### **Time Management:**
- Build in buffer time for technical issues
- Have backup datasets ready
- Prepare pre-written code snippets
- Consider live coding vs showing completed examples
- Balance theory (30%) with practice (70%)

### **Student Engagement:**
- Use poll questions during lectures
- Encourage questions throughout
- Share industry anecdotes
- Show "before and after" EDA examples
- Celebrate student discoveries in data

This plan ensures comprehensive coverage while maintaining a logical progression from basics to advanced topics! 🎓