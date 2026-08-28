##############################################################################
# EXPLORATORY DATA ANALYSIS (EDA) - Medical Students Dataset
# Structure follows the CSE422 EDA Lab notebook step-by-step
#
# Stages of EDA (from lab):
#   1. Descriptive Analysis
#   2. Correlation Analysis
#   3. Check imbalance in data
##############################################################################

# ============================================================================
# Importing Required Python Libraries
# ============================================================================
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create output directory for all graphs
os.makedirs('Data Visuals', exist_ok=True)

# ============================================================================
# Load Dataset
# ============================================================================
dataset = pd.read_csv("medical_students_dataset.csv")

# ============================================================================
# Summarize Data
# Reason: Get a quick overview of the entire dataset — how large it is,
# what the first few rows look like, and the overall shape.
# This is the very first thing we do in any EDA to orient ourselves.
# ============================================================================

# Display the full dataset overview (head + tail)
print("=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)
print(dataset)
print()

# Display the first 10 rows to visually inspect the data
print("First 10 rows:")
print(dataset.head(10))
print()

# Shape of the dataset
print('Shape of the dataset is {}. This dataset contains {} rows and {} columns.'.format(
    dataset.shape, dataset.shape[0], dataset.shape[1]))
print()

# ============================================================================
# Feature Names and its Datatypes
# Reason: Understanding dtypes is critical — it tells us which columns are
# numeric (can be used in math/correlation) and which are categorical
# (need special handling like encoding). Also reveals if any column has an
# unexpected type, e.g. a numeric column read as object due to dirty data.
# ============================================================================
print("=" * 70)
print("FEATURE NAMES AND DATATYPES")
print("=" * 70)
dataset.info()
print()

# ============================================================================
# Data Splitting — Separate Numerical and Categorical Features
# Reason: Numerical and categorical features require different analysis
# techniques. Numerical features use statistics like mean/std/variance,
# while categorical features use counts and frequency analysis.
# Separating them early makes the rest of the EDA cleaner.
# ============================================================================
print("=" * 70)
print("DATA SPLITTING")
print("=" * 70)

# Selecting numerical features
# Reason: We need to isolate numeric columns for statistical summaries,
# correlation analysis, and density plots. Medical measurements like
# Height, Weight, BMI, Temperature, Heart Rate, Blood Pressure, and
# Cholesterol are all numeric and are the core clinical variables.
numerical_data = dataset.select_dtypes(include='number')
numerical_features = numerical_data.columns.tolist()
print(f'There are {len(numerical_features)} numerical features:', '\n')
print(numerical_features)
print()

# Selecting categorical features
# Reason: Categorical columns like Gender, Blood Type, Diabetes, and Smoking
# need frequency-based analysis (value_counts, bar plots) rather than
# mean/std. Identifying them lets us apply the right techniques.
categorical_data = dataset.select_dtypes(include='object')
categorical_features = categorical_data.columns.tolist()
print(f'There are {len(categorical_features)} categorical features:', '\n')
print(categorical_features)
print()

# ============================================================================
# STAGE 1: DESCRIPTIVE ANALYSIS
# In descriptive analysis we analyze each variable separately to get
# inference about the feature.
# ============================================================================
print("=" * 70)
print("STAGE 1: DESCRIPTIVE ANALYSIS")
print("=" * 70)

# Summary statistics of Numerical Features
# Reason: describe() gives us count, mean, std, min, 25%, 50%, 75%, max
# for every numerical column. This immediately reveals:
#   - Missing values (count < total rows)
#   - The central tendency and spread of each clinical measurement
#   - Potential outliers (large gap between 75% and max)
# Transposing makes it easier to read when there are many features.
print("\nSummary statistics of Numerical Features (Transposed):")
print(numerical_data.describe().T)
print()

# Summary statistics of Categorical Features
# Reason: For categorical columns, describe() shows count, unique values,
# the most frequent category (top), and its frequency (freq). This tells us:
#   - If any category dominates (e.g. mostly "Male" or mostly "No" for Smoking)
#   - How many unique categories exist in each feature
#   - Missing value counts (count < total rows)
print("Summary statistics of Categorical Features (Transposed):")
print(categorical_data.describe().T)
print()

# Variance of each numerical feature
# Reason: Variance shows how spread out the data is. Features with very
# low variance (near 0) contribute little information and may be candidates
# for removal. Features with very high variance (like Student ID) may need
# scaling. In this dataset, Student ID will have enormous variance since
# it's just an identifier — this confirms it's not a useful feature.
print("Variance of each numerical feature:")
print(numerical_data.var())
print()

# Number of unique values in each numerical feature
# Reason: Knowing unique counts helps identify:
#   - Identifier columns (Student ID: unique for every row)
#   - Quasi-categorical columns (if a numeric column has very few unique values,
#     it might actually be categorical, e.g. a binary 0/1 flag)
#   - Data granularity (continuous vs discrete)
print("Number of unique values in each numerical feature:")
print(numerical_data.nunique())
print()

# Missing Values in numerical features
# Reason: Missing data directly impacts model training. We need to know
# which features have missing values and how many, so we can decide on
# imputation strategy (mean, median, mode, or drop). Medical data often
# has missing records due to incomplete patient information.
print("Missing values in numerical features:")
print(numerical_data.isnull().sum())
print()

# ============================================================================
# Categorical Features — Detailed Analysis
# ============================================================================
print("=" * 70)
print("CATEGORICAL FEATURES ANALYSIS")
print("=" * 70)

# Number of unique values in each categorical feature
# Reason: This tells us the cardinality of each categorical variable.
# Low cardinality (e.g. Gender: 2, Diabetes: 2, Smoking: 2) means simple
# encoding. Higher cardinality (e.g. Blood Type: 4-8) may need special
# treatment. This also helps verify the data makes sense.
unique_counts = categorical_data.nunique()
print("\nNumber of unique values in each categorical feature:")
print(unique_counts)
print()

# Missing values in categorical features
# Reason: Categorical missing values are common in medical surveys (patients
# may skip questions like Smoking status). We need to know the extent
# of missingness to choose the right imputation (mode, or a new "Unknown" category).
print("Missing values in categorical features:")
print(categorical_data.isnull().sum())
print()

# Barplot of unique value counts in every categorical feature
# Reason: Bar plots visualize the distribution of each categorical variable.
# This immediately shows:
#   - Whether Gender is balanced (equal Male/Female) or skewed
#   - Blood Type distribution — some types are naturally rarer
#   - Diabetes/Smoking prevalence in the student population
# These distributions affect model bias and may need resampling.
for col in categorical_features:
    plt.figure(figsize=(8, 5))
    plt.title(f'Distribution of {col}')
    categorical_data[col].value_counts().sort_index().plot(kind='bar', rot=0, xlabel=col, ylabel='count')
    plt.tight_layout()
    plt.savefig(os.path.join('Data Visuals', f'barplot_{col.replace(" ", "_")}.png'), dpi=150, bbox_inches='tight')
    plt.show()

# ============================================================================
# STAGE 2: CORRELATION ANALYSIS
# ============================================================================
print("=" * 70)
print("STAGE 2: CORRELATION ANALYSIS")
print("=" * 70)

# Correlation matrix of whole dataset (numerical features)
# Reason: The correlation matrix shows pairwise linear relationships between
# all numerical features. For medical data, we expect:
#   - Height and Weight to be positively correlated
#   - Weight and BMI to be strongly correlated (BMI = Weight/Height^2)
#   - Heart Rate and Blood Pressure may show some correlation
# Highly correlated features (multicollinearity) can hurt some models.
correlation_matrix = numerical_data.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)
print()

# Correlation Heatmap plot
# Reason: A heatmap makes it much easier to spot strong correlations visually.
# The color intensity immediately highlights which pairs of features are
# strongly related. The annotation shows exact values for precision.
# Using 'coolwarm' colormap so positive correlations are red and negative are blue.
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.3f', linewidths=0.3)
plt.title('Correlation Heatmap of Numerical Features')
plt.tight_layout()
plt.savefig(os.path.join('Data Visuals', 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.show()

# Generating correlation plot between features and target variable using
# different methods
# Reason: Different correlation methods capture different types of relationships:
#   - Pearson: measures LINEAR relationships (best for normally distributed data)
#   - Spearman: measures MONOTONIC relationships using ranks (robust to outliers)
#   - Kendall: also measures ordinal association (more conservative, good for ties)
# Since medical data may not be perfectly linear (e.g. Age vs Cholesterol may
# plateau), using multiple methods gives a more complete picture.
# We correlate against Cholesterol as a key health indicator since the dataset
# has no explicit target variable — Cholesterol is the most clinically relevant
# continuous outcome to investigate risk factors for.

# Drop Student ID from correlation — it's just an identifier, not a feature.
# Including it would produce meaningless correlation values.
numerical_data_no_id = numerical_data.drop(columns=['Student ID'], errors='ignore')

fig, ax = plt.subplots(3, 1, figsize=(10, 12))

# Correlation coefficient using different methods against Cholesterol
# Reason: Cholesterol is chosen as the reference variable because it's a major
# health risk indicator. Understanding which features correlate with it
# (e.g. does BMI or Age drive higher cholesterol?) is medically important.
corr1 = numerical_data_no_id.corr('pearson')[['Cholesterol']].sort_values(by='Cholesterol', ascending=False)
corr2 = numerical_data_no_id.corr('spearman')[['Cholesterol']].sort_values(by='Cholesterol', ascending=False)
corr3 = numerical_data_no_id.corr('kendall')[['Cholesterol']].sort_values(by='Cholesterol', ascending=False)

# Setting titles for each plot
ax[0].set_title('Pearson method')
ax[1].set_title('Spearman method')
ax[2].set_title('Kendall method')

# Generating heatmaps of each method
sns.heatmap(corr1, ax=ax[0], annot=True, cmap='coolwarm')
sns.heatmap(corr2, ax=ax[1], annot=True, cmap='coolwarm')
sns.heatmap(corr3, ax=ax[2], annot=True, cmap='coolwarm')

plt.suptitle('Correlation with Cholesterol using Different Methods', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join('Data Visuals', 'correlation_methods.png'), dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# STAGE 3: CHECK IMBALANCE IN DATA
# We check the balance of categorical target-like variables since this is
# relevant for any classification task we might build later.
# ============================================================================
print("=" * 70)
print("STAGE 3: CHECK IMBALANCE IN DATA")
print("=" * 70)

# Check imbalance in Diabetes
# Reason: If we want to predict Diabetes (Yes/No), the classes must be
# reasonably balanced. A highly imbalanced dataset (e.g. 95% No, 5% Yes)
# would make a naive classifier that always predicts "No" appear accurate.
# We need to quantify the imbalance to decide if resampling is needed.
print("\n--- Diabetes Imbalance ---")
diabetes_counts = dataset.groupby("Diabetes").size()
diabetes_classes = diabetes_counts.index.tolist()
total_diabetes = diabetes_counts.sum()

columns = ['class', 'count', 'percentage']
d_count = []
d_percentage = []

for cls in diabetes_classes:
    d_count.append(diabetes_counts[cls])
    percent = (diabetes_counts[cls] / total_diabetes) * 100
    d_percentage.append(round(percent, 2))

imbalance_diabetes_df = pd.DataFrame(
    list(zip(diabetes_classes, d_count, d_percentage)), columns=columns)
print(imbalance_diabetes_df)
print()

# Barplot of Diabetes vs Percentage
# Reason: Visualizing the class split makes imbalance immediately obvious.
# A bar chart clearly shows if one class dominates the other.
plt.figure(figsize=(6, 4))
sns.barplot(data=imbalance_diabetes_df, x='class', y='percentage')
plt.title('Diabetes Class Distribution (%)')
plt.xlabel('Diabetes')
plt.ylabel('Percentage (%)')
plt.tight_layout()
plt.savefig(os.path.join('Data Visuals', 'imbalance_diabetes.png'), dpi=150, bbox_inches='tight')
plt.show()

# Check imbalance in Smoking
# Reason: Same logic as Diabetes — if Smoking is a prediction target,
# we need balanced classes. Even if it's a feature, understanding its
# distribution helps interpret the dataset (e.g., is this a population
# with mostly non-smokers?).
print("--- Smoking Imbalance ---")
smoking_counts = dataset.groupby("Smoking").size()
smoking_classes = smoking_counts.index.tolist()
total_smoking = smoking_counts.sum()

s_count = []
s_percentage = []

for cls in smoking_classes:
    s_count.append(smoking_counts[cls])
    percent = (smoking_counts[cls] / total_smoking) * 100
    s_percentage.append(round(percent, 2))

imbalance_smoking_df = pd.DataFrame(
    list(zip(smoking_classes, s_count, s_percentage)), columns=columns)
print(imbalance_smoking_df)
print()

# Barplot of Smoking vs Percentage
plt.figure(figsize=(6, 4))
sns.barplot(data=imbalance_smoking_df, x='class', y='percentage')
plt.title('Smoking Class Distribution (%)')
plt.xlabel('Smoking')
plt.ylabel('Percentage (%)')
plt.tight_layout()
plt.savefig(os.path.join('Data Visuals', 'imbalance_smoking.png'), dpi=150, bbox_inches='tight')
plt.show()



# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
Performed various exploratory data analysis techniques on the Medical
Students Dataset:

1. DESCRIPTIVE ANALYSIS:
   - Examined 13 features (9 numerical, 4 categorical)
   - Identified missing values across multiple columns
   - Computed summary statistics, variance, and unique value counts

2. CORRELATION ANALYSIS:
   - Generated full correlation matrix and heatmap
   - Compared Pearson, Spearman, and Kendall correlation methods
     against Cholesterol as the key health indicator
   - Identified relationships between clinical measurements

3. IMBALANCE CHECK:
   - Analyzed class distribution for Diabetes and Smoking
   - Quantified percentage split to assess if resampling is needed

4. All generated graphs are saved to the 'Data Visuals/' folder.

From these insights, we can make informed decisions about:
   - Data cleaning (handle missing values, remove/cap outliers)
   - Feature engineering (drop Student ID, potential interactions)
   - Model selection (consider class imbalance techniques if needed)
   - Preprocessing (scaling, encoding categorical variables)
""")
