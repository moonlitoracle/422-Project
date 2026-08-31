import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ============================================================================
# 1. Load Original Dataset
# ============================================================================
df_raw = pd.read_csv('medical_students_dataset.csv')
df = df_raw.copy()

# ============================================================================
# 2. Drop Irrelevant Identifier
# ============================================================================
if 'Student ID' in df.columns:
    df = df.drop(columns=['Student ID'])

# ============================================================================
# 3. Drop Target Missing Values
# ============================================================================
df = df.dropna(subset=['Diabetes'])

# ============================================================================
# 4. Impute Missing Values
# ============================================================================
# Fill missing BMI using Height and Weight
bmi_mask = df['BMI'].isnull() & df['Height'].notnull() & df['Weight'].notnull()
df.loc[bmi_mask, 'BMI'] = df.loc[bmi_mask, 'Weight'] / ((df.loc[bmi_mask, 'Height'] / 100) ** 2)

# Fill missing Weight using Height and BMI
weight_mask = df['Weight'].isnull() & df['Height'].notnull() & df['BMI'].notnull()
df.loc[weight_mask, 'Weight'] = df.loc[weight_mask, 'BMI'] * ((df.loc[weight_mask, 'Height'] / 100) ** 2)

# Fill missing Height using Weight and BMI
height_mask = df['Height'].isnull() & df['Weight'].notnull() & df['BMI'].notnull()
df.loc[height_mask, 'Height'] = 100 * np.sqrt(df.loc[height_mask, 'Weight'] / df.loc[height_mask, 'BMI'])

num_cols = ['Age', 'Height', 'Weight', 'BMI', 'Temperature', 'Heart Rate', 'Blood Pressure', 'Cholesterol']
cat_cols = ['Gender', 'Blood Type', 'Smoking']

# Impute Numerical Features with Median
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Impute Categorical Features with Mode
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# ============================================================================
# 5. Outlier Treatment using IQR Capping (Winsorization)
# ============================================================================
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df[col] = np.clip(df[col], lower_bound, upper_bound)

# ============================================================================
# 6. Categorical Encoding
# ============================================================================
le = LabelEncoder()
df['Diabetes'] = le.fit_transform(df['Diabetes'])
df['Gender'] = le.fit_transform(df['Gender'])
df['Smoking'] = le.fit_transform(df['Smoking'])

# One-Hot Encoding for Blood Type (keeps values as 0 and 1)
df = pd.get_dummies(df, columns=['Blood Type'], drop_first=True, dtype=int)

# ============================================================================
# 7. Export Processed Dataset (BEFORE Scaling)
#    Scaling is intentionally excluded here to prevent data leakage.
#    The scaler must be fit on the training set only (see Step 9).
# ============================================================================
df.to_csv('processed.csv', index=False)
print("Exported: processed.csv")

# ============================================================================
# 8. Stratified Train-Test Split (80/20)
#    stratify=y preserves the Diabetes class ratio in both sets
# ============================================================================
X = df.drop(columns=['Diabetes'])
y = df['Diabetes']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================================
# 9. Feature Scaling (AFTER Split — No Data Leakage)
#    The scaler is fit ONLY on training data, then used to transform both
#    train and test sets. This prevents test data statistics from leaking
#    into the model during training.
# ============================================================================
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# ============================================================================
# 10. Recombine & Export Train and Test Sets
# ============================================================================
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)

print(f"Exported: train.csv ({train_df.shape[0]} rows)")
print(f"Exported: test.csv  ({test_df.shape[0]} rows)")

# Verification: Diabetes distribution preserved
print("\nDiabetes distribution (proportion):")
print(f"  Processed: {y.value_counts(normalize=True).to_dict()}")
print(f"  Train:     {y_train.value_counts(normalize=True).to_dict()}")
print(f"  Test:      {y_test.value_counts(normalize=True).to_dict()}")

print("\nPreprocessing & splitting complete!")
