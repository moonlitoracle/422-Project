import pandas as pd
from sklearn.model_selection import train_test_split

# Load preprocessed dataset
df = pd.read_csv('processed.csv')

# Define features and target
X = df.drop(columns=['Diabetes'])
y = df['Diabetes']

# Stratified split: 80% training, 20% testing
# stratify=y ensures the Diabetes class ratio is preserved in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Recombine features and target for export
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# Export to CSV
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)

# Verification
print("Split complete!\n")
print(f"Training set: {train_df.shape[0]} rows")
print(f"Test set:     {test_df.shape[0]} rows\n")
print("Diabetes distribution (proportion):")
print(f"  Processed: {y.value_counts(normalize=True).to_dict()}")
print(f"  Train:     {y_train.value_counts(normalize=True).to_dict()}")
print(f"  Test:      {y_test.value_counts(normalize=True).to_dict()}")
