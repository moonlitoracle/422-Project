import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Load original dataset safely
df_raw = pd.read_csv('medical_students_dataset.csv')
df = df_raw.copy()

# Step 1: Drop Irrelevant Identifier
if 'Student ID' in df.columns:
    df = df.drop(columns=['Student ID'])

# Step 2: Drop Target Missing Values
df = df.dropna(subset=['Diabetes'])

# Step 3: Impute Missing Values
#Fill missing BMI using Height and Weight
bmi_mask = df['BMI'].isnull() & df['Height'].notnull() & df['Weight'].notnull()
df.loc[bmi_mask, 'BMI'] = df.loc[bmi_mask, 'Weight'] / ((df.loc[bmi_mask, 'Height'] / 100) ** 2)

#Fill missing Weight using Height and BMI
weight_mask = df['Weight'].isnull() & df['Height'].notnull() & df['BMI'].notnull()
df.loc[weight_mask, 'Weight'] = df.loc[weight_mask, 'BMI'] * ((df.loc[weight_mask, 'Height'] / 100) ** 2)

#Fill missing Height using Weight and BMI
height_mask = df['Height'].isnull() & df['Weight'].notnull() & df['BMI'].notnull()
df.loc[height_mask, 'Height'] = 100 * np.sqrt(df.loc[height_mask, 'Weight'] / df.loc[height_mask, 'BMI'])

num_cols = ['Age', 'Height', 'Weight', 'BMI', 'Temperature', 'Heart Rate', 'Blood Pressure', 'Cholesterol']
cat_cols = ['Gender', 'Blood Type', 'Smoking']

#Impute Numerical Features with Median
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Impute Categorical Features with Mode
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Outlier Treatment using IQR Capping (Winsorization)
# ---------------------------------------------------------
# IQR capping limits extreme values to 1.5 * IQR bounds
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df[col] = np.clip(df[col], lower_bound, upper_bound)

# Step 4: Categorical Encoding
le = LabelEncoder()
df['Diabetes'] = le.fit_transform(df['Diabetes'])
df['Gender'] = le.fit_transform(df['Gender'])
df['Smoking'] = le.fit_transform(df['Smoking'])

# One-Hot Encoding for Blood Type (keeps values as 0 and 1)
df = pd.get_dummies(df, columns=['Blood Type'], drop_first=True, dtype=int)

# ---------------------------------------------------------
# Step 5: Feature Scaling (ONLY Continuous Variables)
# ---------------------------------------------------------
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Step 6: Export updated preprocessed CSV
output_filename = 'processed.csv'
df.to_csv(output_filename, index=False)

print("Preprocessing complete!")