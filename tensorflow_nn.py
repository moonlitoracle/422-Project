##############################################################################
# NEURAL NETWORK — TensorFlow / Keras Implementation
# Follows the TensorFlow section of CSE422_NeuralNetworkLab (Cells 28–38)
#
# Architecture: Dense(64, relu) → Dense(32, relu) → Dense(1, sigmoid)
# Loss:         BinaryCrossentropy (binary classification: Diabetes 0/1)
# Optimizer:    Adam (lr=0.001)
##############################################################################

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

# Create output directory for visuals
os.makedirs('Model Visuals', exist_ok=True)

# ============================================================================
# 1. Load Train & Test Sets
# ============================================================================
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

X_train = train_df.drop(columns=['Diabetes']).values
y_train = train_df['Diabetes'].values
X_test = test_df.drop(columns=['Diabetes']).values
y_test = test_df['Diabetes'].values

print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"Test set:     {X_test.shape[0]} samples, {X_test.shape[1]} features\n")

# ============================================================================
# 2. Build the Model (Lab Cell 36 pattern)
# ============================================================================
# Lab used: Flatten → Dense(128, relu) → Dense(10, softmax) for MNIST (10 classes)
# We adapt: Dense(64, relu) → Dense(32, relu) → Dense(1, sigmoid) for binary
# classification (Diabetes yes/no). Sigmoid outputs a probability between 0–1.
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# ============================================================================
# 3. Compile the Model (Lab Cell 36 pattern)
# ============================================================================
# Lab used: Adam(0.001), SparseCategoricalCrossentropy (multi-class)
# We use:   Adam(0.001), BinaryCrossentropy (binary classification)
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=BinaryCrossentropy(),
    metrics=['accuracy']
)

model.summary()

# ============================================================================
# 4. Train the Model (Lab Cell 36 pattern)
# ============================================================================
# Compute class weights to handle imbalanced data (same idea as
# class_weight='balanced' in Logistic Regression). Without this, the model
# predicts "No Diabetes" for every sample because ~90% of the data is class 0.
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}
print(f"Class weights: {class_weight_dict}\n")

# validation_split=0.1 holds out 10% of training data to monitor overfitting
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=128,
    validation_split=0.1,
    class_weight=class_weight_dict,
    verbose=1
)

# ============================================================================
# 5. Plot Loss Curve (Lab Cell 38 pattern)
# ============================================================================
# Lab plotted training loss only; we also add validation loss to check for
# overfitting (validation loss rising while training loss falls = overfitting)
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--')
plt.title('TensorFlow Neural Network — Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Model Visuals/tf_loss_curve.png', dpi=150)
plt.close()
print("\nSaved: tf_loss_curve.png")

# ============================================================================
# 6. Evaluate & Print Results
# ============================================================================
# Get probability predictions and convert to binary using 0.5 threshold
y_prob = model.predict(X_test)
y_pred = (y_prob > 0.5).astype(int).flatten()

# Test accuracy (same as lab Cell 38)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTesting Accuracy: {round(test_acc * 100, 2)}%")

# Precision, Recall, F1 (same metrics as models.py for comparison)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n--- Neural Network (TensorFlow) ---")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1 Score:  {f1:.4f}")

print("\nTensorFlow Neural Network complete!")
