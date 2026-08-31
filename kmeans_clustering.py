##############################################################################
# K-MEANS CLUSTERING (Unsupervised Learning)
# Treats the dataset as an unsupervised problem as required by the
# CSE422 Project Template.
#
# Structure follows the K-Means Lab:
#   1. Load & prepare data (drop target — unsupervised)
#   2. Elbow Method to find optimal K
#   3. Apply K-Means with chosen K
#   4. Visualize clusters using PCA (2D projection)
#   5. Compare cluster assignments to actual Diabetes labels
##############################################################################

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Create output directory for cluster visuals
os.makedirs('Model Visuals', exist_ok=True)

# ============================================================================
# 1. Load Processed Dataset & Prepare Features
# ============================================================================
# K-Means is unsupervised, so we use the processed dataset (not the
# train/test split) and DROP the target variable (Diabetes).
# The model must find structure in the data WITHOUT knowing the labels.
df = pd.read_csv('processed.csv')

# Separate features and save the true labels for comparison later
y_true = df['Diabetes']
X = df.drop(columns=['Diabetes'])

# Scale the features — processed.csv is unscaled (scaling was moved to
# after the train/test split for supervised models). K-Means is distance-
# based, so features must be on the same scale to prevent high-variance
# features from dominating cluster assignment.
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

print(f"Dataset: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
print(f"True Diabetes distribution: {y_true.value_counts().to_dict()}\n")

# ============================================================================
# 2. Elbow Method — Finding the Optimal Number of Clusters
# ============================================================================
# WHY: In unsupervised learning we don't know how many groups exist in the
# data. The Elbow Method tests different values of K and plots the SSE
# (Sum of Squared Errors, also called inertia) — the total distance of
# each point to its assigned cluster centroid.
#
# As K increases, SSE always decreases. The "elbow" is the point where
# adding more clusters gives diminishing returns, indicating the natural
# number of groups in the data.
#
# Lab Reference: Cells 35–38 of CSE_422_Lab_5_part_1_[K_Means_Clustering]

k_range = range(1, 10)
SSE = []

print("Running Elbow Method (K = 1 to 9)...")
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    SSE.append(km.inertia_)
    print(f"  K={k}: SSE = {km.inertia_:.2f}")

# Plot K vs SSE (Elbow Curve)
# Lab Reference: Cell 38
plt.figure(figsize=(8, 5))
plt.plot(list(k_range), SSE, 'bo-', linewidth=2, markersize=8)
plt.xlabel('K (Number of Clusters)')
plt.ylabel('Sum of Squared Error (SSE / Inertia)')
plt.title('Elbow Method — Optimal K Selection')
plt.xticks(list(k_range))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Model Visuals/elbow_method.png', dpi=150)
plt.close()
print("\nSaved: elbow_method.png")

# ============================================================================
# 3. Apply K-Means with Chosen K
# ============================================================================
# Based on the Elbow plot and domain knowledge (2 classes: Diabetes/No
# Diabetes), we use K=2.
# Lab Reference: Cells 20–21, 29, 31

chosen_k = 2
print(f"\nApplying K-Means with K={chosen_k}...")

km_final = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
y_predicted = km_final.fit_predict(X_scaled)

# Add cluster assignments to the dataframe
df['Cluster'] = y_predicted

print(f"\nCluster Centers (shape: {km_final.cluster_centers_.shape}):")
print(pd.DataFrame(km_final.cluster_centers_, columns=X.columns).to_string())

# ============================================================================
# 4. Cluster Distribution
# ============================================================================
# Lab Reference: Cells 27, 33–34
print("\n--- Cluster Distribution ---")
print(f"Predicted Cluster Distribution:\n{df['Cluster'].value_counts().to_string()}\n")

print("Original Diabetes Distribution:")
print(f"{y_true.value_counts().to_string()}\n")

# Cross-tabulation: How do clusters map to actual Diabetes labels?
cross_tab = pd.crosstab(df['Cluster'], y_true, margins=True)
print("Cluster vs Actual Diabetes (Cross-Tabulation):")
print(cross_tab.to_string())

# ============================================================================
# 5. Visualize Clusters using PCA (2D Projection)
# ============================================================================
# Since we have 13 features, we cannot plot them directly. PCA reduces
# the data to 2 dimensions while preserving the most variance, allowing
# us to visualize the clusters on a 2D scatter plot.
# Lab Reference: Cells 23–24, 30–31 (lab used 2 features directly;
# we use PCA because we have 13 features)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"\nPCA explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.4f}")

# Transform cluster centers to PCA space for plotting
centers_pca = pca.transform(km_final.cluster_centers_)

# Plot: Before vs After Clustering (side-by-side)
# Lab Reference: Cell 31 (before/after comparison)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Before clustering (all points same color)
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], color='gray', alpha=0.3, s=10)
axes[0].set_title('Before Clustering')
axes[0].set_xlabel('PCA Component 1')
axes[0].set_ylabel('PCA Component 2')

# After clustering (colored by cluster assignment)
colors = ['#3498db', '#e74c3c']
for i in range(chosen_k):
    mask = y_predicted == i
    axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                    color=colors[i], alpha=0.3, s=10, label=f'Cluster {i}')
# Plot centroids
axes[1].scatter(centers_pca[:, 0], centers_pca[:, 1],
                color='black', marker='X', s=200, edgecolors='white',
                linewidths=2, label='Centroids', zorder=5)
axes[1].set_title('After K-Means Clustering (K=2)')
axes[1].set_xlabel('PCA Component 1')
axes[1].set_ylabel('PCA Component 2')
axes[1].legend()

plt.tight_layout()
plt.savefig('Model Visuals/kmeans_clusters.png', dpi=150)
plt.close()
print("\nSaved: kmeans_clusters.png")

# Plot: Clusters vs Actual Diabetes Labels (side-by-side comparison)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# K-Means clusters
for i in range(chosen_k):
    mask = y_predicted == i
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                    color=colors[i], alpha=0.3, s=10, label=f'Cluster {i}')
axes[0].scatter(centers_pca[:, 0], centers_pca[:, 1],
                color='black', marker='X', s=200, edgecolors='white',
                linewidths=2, zorder=5)
axes[0].set_title('K-Means Clusters')
axes[0].set_xlabel('PCA Component 1')
axes[0].set_ylabel('PCA Component 2')
axes[0].legend()

# Actual Diabetes labels
label_colors = ['#2ecc71', '#e74c3c']
label_names = ['No Diabetes', 'Diabetes']
for i in range(2):
    mask = y_true == i
    axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                    color=label_colors[i], alpha=0.3, s=10, label=label_names[i])
axes[1].set_title('Actual Diabetes Labels')
axes[1].set_xlabel('PCA Component 1')
axes[1].set_ylabel('PCA Component 2')
axes[1].legend()

plt.tight_layout()
plt.savefig('Model Visuals/kmeans_vs_actual.png', dpi=150)
plt.close()
print("Saved: kmeans_vs_actual.png")

print("\nK-Means clustering complete!")
