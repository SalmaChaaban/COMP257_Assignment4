# %%
# Salma Chaaban -301216551
# COMP257 - Assignment 4

# %%
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from scipy.ndimage import rotate
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# %%
# 1. Retrieve and load the Olivetti faces dataset
olivetti_faces = fetch_olivetti_faces()

X = olivetti_faces.data  #  flattened 1D array format
y = olivetti_faces.target  # The target labels

X.shape

# %%
# 2. Split the training set, a validation set, and a test set using stratified sampling to ensure that there are the same number of images per person in each set

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=51)
for train_idx, temp_idx in sss.split(X, y):
    X_train, X_temp = X[train_idx], X[temp_idx]
    y_train, y_temp = y[train_idx], y[temp_idx]


sss_val_test = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=51)
for val_idx, test_idx in sss_val_test.split(X_temp, y_temp):
    X_val, X_test = X_temp[val_idx], X_temp[test_idx]
    y_val, y_test = y_temp[val_idx], y_temp[test_idx]

# %%
X_train.shape

# %%
# 3. Apply PCA on the training data to preserve 99% of the variance

pca = PCA(n_components=0.99, random_state=51)
X_train_pca = pca.fit_transform(X_train)

n_components = pca.n_components_
print(f'Number of PCA components: {n_components}')
X_train_pca.shape

# %%
# 4. Determine the most suitable covariance type for the dataset

covariance_types = ['full', 'tied', 'diag', 'spherical']
results = {
    'Covariance Type': [],
    'Number of Clusters': [],
    'AIC': [],
    'BIC': []
}

# Loop through each covariance type and number of clusters
for cov_type in covariance_types:
    for n_clusters in range(2, 32):  # Try between 2 and 31 clusters
        gmm = GaussianMixture(n_components=n_clusters, covariance_type=cov_type, random_state=51)
        gmm.fit(X_train_pca)
        aic = gmm.aic(X_train_pca)
        bic = gmm.bic(X_train_pca)
        
        # Append results
        results['Covariance Type'].append(cov_type)
        results['Number of Clusters'].append(n_clusters)
        results['AIC'].append(aic)
        results['BIC'].append(bic)

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Print all results for comparison
print(results_df)

# Optionally, print the best results
best_aic_row = results_df.loc[results_df['AIC'].idxmin()]
best_bic_row = results_df.loc[results_df['BIC'].idxmin()]

print(f'\nBest covariance type based on AIC: {best_aic_row["Covariance Type"]} (AIC={best_aic_row["AIC"]}, Clusters={best_aic_row["Number of Clusters"]})')
print(f'Best covariance type based on BIC: {best_bic_row["Covariance Type"]} (BIC={best_bic_row["BIC"]}, Clusters={best_bic_row["Number of Clusters"]})')

# Plot AIC and BIC
plt.figure(figsize=(12, 6))
for cov_type in covariance_types:
    subset = results_df[results_df['Covariance Type'] == cov_type]
    plt.plot(subset['Number of Clusters'], subset['AIC'], label=f'AIC - {cov_type}')
    plt.plot(subset['Number of Clusters'], subset['BIC'], label=f'BIC - {cov_type}', linestyle='--')

plt.xlabel('Number of Clusters')
plt.ylabel('Information Criterion Value')
plt.title('AIC and BIC Comparison')
plt.legend()
plt.grid()
plt.show()


# %%
# 5. Determine the minimum number of clusters that best represent the dataset using either AIC or BIC

# Determine the minimum number of clusters based on AIC
min_aic_row = results_df.loc[results_df['AIC'].idxmin()]
min_aic_clusters = min_aic_row['Number of Clusters']
min_aic_value = min_aic_row['AIC']

# Determine the minimum number of clusters based on BIC
min_bic_row = results_df.loc[results_df['BIC'].idxmin()]
min_bic_clusters = min_bic_row['Number of Clusters']
min_bic_value = min_bic_row['BIC']

# Print the results
print(f'Minimum number of clusters based on AIC: {min_aic_clusters} (AIC={min_aic_value})')
print(f'Minimum number of clusters based on BIC: {min_bic_clusters} (BIC={min_bic_value})') 

# %%
# 6. Plot the results from steps 3 and 4

# Plot cumulative explained variance, step 3
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.title('PCA - Explained Variance')
plt.show()

n_clusters_range = range(2, 32)
aic_scores = []

# Calculate AIC scores for different cluster sizes, step 4
for n_clusters in n_clusters_range:
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=51)
    gmm.fit(X_train_pca)
    aic_scores.append(gmm.aic(X_train_pca))

# Plot AIC scores
plt.plot(n_clusters_range, aic_scores)
plt.xlabel('Number of clusters')
plt.ylabel('AIC Score')
plt.title('AIC vs Number of Clusters')
plt.show()

# %%
# 7. Output the hard clustering assignments for each instance to identify which cluster each image belongs to

# Fit the final GMM model
gmm_final = GaussianMixture(n_components=6, covariance_type='full', random_state=51)
gmm_final.fit(X_train_pca)

# Get the hard clustering labels
hard_assignments = gmm_final.predict(X_train_pca)
print("Hard clustering assignments:", hard_assignments)

# %%
X_train_pca_2d = X_train_pca[:, :2]

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_train_pca_2d[:, 0], X_train_pca_2d[:, 1], c=hard_assignments, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Cluster Assignment')
plt.title('PCA - Clustering Assignments')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid()
plt.show()

# %%
# Number of clusters to visualize
n_clusters = 6

images = olivetti_faces.images

plt.figure(figsize=(18, 12))

# Loop through each cluster
for cluster in range(n_clusters):
    # Get the indices of the images belonging to the current cluster
    cluster_indices = np.where(hard_assignments == cluster)[0]
    
    # Select a maximum of 10 images from the current cluster for visualization
    for i, idx in enumerate(cluster_indices[:10]):
        plt.subplot(n_clusters, 10, cluster * 10 + i + 1)
        plt.imshow(images[idx], cmap='gray')  # Assuming images are in a variable called 'images'
        plt.axis('off')
        plt.title(f'Cluster {cluster}')
        
plt.tight_layout()
plt.show()

# %%
# 8. Output the soft clustering probabilities for each instance to show the likelihood of each image belonging to each cluster

# Get the soft clustering probabilities (probability of each image belonging to each cluster)
soft_assignments = gmm_final.predict_proba(X_train_pca)
print("Soft clustering probabilities (first instance):", soft_assignments)

# %%
# 9. Use the model to generate some new faces (using sample()) and visualize them (use inverse_transform() to transform the data back to its original space based on the PCA method used

# Generate new faces
new_faces, _ = gmm_final.sample(10)

# Inverse transform to original space
faces_original = pca.inverse_transform(new_faces)

# Visualize generated faces
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(faces_original[i].reshape(64, 64), cmap='gray')
    plt.axis('off')
    
plt.show()

# %%
# 10. Modify some images (e.g., rotate, flip, darken)

rotation_angle = 90  # Rotate sideways
darken_factor = 0.5  # Reduce brightness by 50%

# Rotate and darken each image
modified_images = []
for img in faces_original:
    # Rotate the image by the specified angle
    rotated_img = rotate(img.reshape(64, 64), angle=rotation_angle)

    # Darken the image by multiplying it by the darken factor
    darkened_img = rotated_img * darken_factor

    # Clip values to ensure they remain in valid range [0, 1]
    modified_images.append(np.clip(darkened_img, 0, 1))

# Plot the modified images
plt.figure(figsize=(12, 6))
for i, img in enumerate(modified_images[:10]):
    plt.subplot(2, 5, i + 1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
plt.suptitle('Modified Generated Faces')
plt.show()

# %%
# 11. Determine if the model can detect the anomalies produced in step 10 by comparing the output of the score_samples() method for normal images and for anomalies

# Calculate log-likelihood scores for normal images
normal_scores = gmm_final.score_samples(X_train_pca)

# Calculate log-likelihood scores for modified images (anomalies)
# Flatten the modified images and apply PCA before scoring
modified_images_flattened = [img.flatten() for img in modified_images]
modified_images_pca = pca.transform(modified_images_flattened)
anomaly_scores = gmm_final.score_samples(modified_images_pca)

# Print and compare the scores
print("Log-likelihood scores for normal images (first 10):", normal_scores[:10])
print("Log-likelihood scores for anomalies (first 10):", anomaly_scores[:10])

# Visualization of scores
plt.figure(figsize=(10, 5))
plt.plot(normal_scores[:20], label="Normal Images", marker='o')
plt.plot(anomaly_scores[:20], label="Anomalies", marker='o')
plt.xlabel("Image Index")
plt.ylabel("Log-Likelihood Score")
plt.title("Log-Likelihood Scores for Normal Images vs. Anomalies")
plt.legend()
plt.show()


