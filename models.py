##############################################################################
# MODEL TRAINING & TESTING - Logistic Regression, KNN & Neural Network
##############################################################################

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

# ============================================================================
# 1. Load Train & Test Sets
# ============================================================================
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

X_train = train_df.drop(columns=['Diabetes'])
y_train = train_df['Diabetes']
X_test = test_df.drop(columns=['Diabetes'])
y_test = test_df['Diabetes']

print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"Test set:     {X_test.shape[0]} samples, {X_test.shape[1]} features\n")

# ============================================================================
# 2. Train Models
# ============================================================================
# class_weight='balanced' automatically adjusts weights inversely proportional
# to class frequencies, so the minority class (Diabetes=1) gets a higher weight.
# Without this, Logistic Regression ignores the minority class entirely and
# predicts "No Diabetes" for every sample, resulting in 0 precision/recall.
models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, random_state=42, class_weight='balanced'
    ),
    'KNN (K=3)': KNeighborsClassifier(n_neighbors=3),
    'Neural Network (Sklearn)': MLPClassifier(
        hidden_layer_sizes=(64, 32), activation='relu',
        max_iter=1000, random_state=42
    )
}

# ============================================================================
# 3. Evaluate & Print Results
# ============================================================================
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}\n")
