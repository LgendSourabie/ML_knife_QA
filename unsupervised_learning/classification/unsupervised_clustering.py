#%% IMPORTS
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, silhouette_score, make_scorer
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linear_sum_assignment
import matplotlib.patches as mpatches
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

#%% PREPARE DATA

# Load Knive dataset
file_path = '../../data/chiefs_knife_dataset.xlsx'
df = pd.read_excel(file_path)

# Data cleaning and feature selection
df.dropna(inplace=True)
df_numeric = df.drop(columns=["Number", "Original_Linienanzahl", 'Ra', 'Rz', 'Rq', 'Rt', "Gloss", 'Name', 'Linie', 'Ra_ganz', 'Rq_ganz', 'Rz_ganz', 'Rt_ganz',
       'Ra_Messlinienlange', 'Rq_Messlinienlange', 'Rz_Messlinienlange',
       'Rt_Messlinienlange', 'Ra_ohneRand', 'Rq_ohneRand', 'Rz_ohneRand',
       'Rt_ohneRand'])

# DATA STANDARDIZATION
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# Generate true labels
LOWER_SPECIFICATION_LIMIT = 0.125
UPPER_SPECIFICATION_LIMIT = 0.215
is_between_specification_bounds = (df['Ra'] >= LOWER_SPECIFICATION_LIMIT) & (df['Ra'] < UPPER_SPECIFICATION_LIMIT)
good_product_range = np.where(is_between_specification_bounds, 1, 0)
df.insert(df.columns.get_loc('Ra') + 1, 'in_specification_limit', good_product_range)
y_true = df['in_specification_limit']






#%% FUNCTIONS

# custim silhouette scorer
def silhouette_scorer(estimator, X):
    cluster_labels = estimator.fit_predict(X)
    if len(set(cluster_labels)) > 1:  # At least two clusters
        return silhouette_score(X, cluster_labels)
    else:
        return -1  # Return a low score if only one cluster is found

# convert noise into closest cluster
def assign_noise_to_nearest_cluster(X, labels):
    # Find the centroids of the clusters (excluding noise)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]  # Exclude the noise label (-1)
    
    # Calculate centroids of the existing clusters
    centroids = np.array([X[labels == label].mean(axis=0) for label in unique_labels])
    
    # Find noise points
    noise_indices = np.where(labels == -1)[0]
    
    # Assign each noise point to the nearest centroid
    if noise_indices.size > 0:
        closest_centroids, _ = pairwise_distances_argmin_min(X[noise_indices], centroids)
        labels[noise_indices] = unique_labels[closest_centroids]
    
    return labels


# clustering with grid search
def grid_search_clustering(X):
    results = {}
    best_params = {}
    
    # K-Means
    param_grid = {'n_clusters': [2]}
    kmeans = KMeans(random_state=42)
    grid_search = GridSearchCV(kmeans, param_grid, cv=3, scoring=silhouette_scorer)
    grid_search.fit(X)
    best_kmeans = grid_search.best_estimator_
    kmeans_labels = best_kmeans.predict(X)
    results['K-Means'] = kmeans_labels
    best_params['K-Means'] = grid_search.best_params_
    



    # DBSCAN with noise assignment to nearest cluster
    best_silhouette = -1
    best_dbscan_labels = None
    # Expand the range of parameters for `eps` and `min_samples`
    for eps in np.arange(0.1, 3.0, 0.1):  # Wider range for `eps`
        for min_samples in range(2, 10):  # Wider range for `min_samples`
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            dbscan_labels = dbscan.fit_predict(X)
            num_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)  # Exclude noise points
            
            if num_clusters == 2:  # Focus on cases where exactly 2 clusters are found
                # Assign noise points to the nearest cluster
                dbscan_labels = assign_noise_to_nearest_cluster(X, dbscan_labels)
                
                score = silhouette_score(X, dbscan_labels)
                if score > best_silhouette:
                    best_silhouette = score
                    best_dbscan_labels = dbscan_labels
                    best_params['DBSCAN'] = {'eps': eps, 'min_samples': min_samples}
    results['DBSCAN'] = best_dbscan_labels


    
    # Gaussian Mixture Models
    param_grid = {
        'n_components': [2],  # We only want 2 clusters
        'covariance_type': ['full'],  # Testing different covariance types
        'init_params': ['kmeans', 'random'],  # Different initialization methods
        'max_iter': [100, 200, 300],  # Different numbers of maximum iterations
        'n_init': [1, 5, 10]  # Number of initializations
    }
    gmm = GaussianMixture(random_state=42)
    grid_search = GridSearchCV(gmm, param_grid, cv=3, scoring=silhouette_scorer)
    grid_search.fit(X)
    best_gmm = grid_search.best_estimator_
    gmm_labels = best_gmm.predict(X)
    results['GMM'] = gmm_labels
    best_params['GMM'] = grid_search.best_params_
    
    return results, best_params

# visualize generated clusters
def visualize_clusters(X, true_labels, clustering_results):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))  # Adjusted for 4 plots
    
    # Define custom colors for the labels
    colors = {1: '#90EE90', 0: '#FFA07A', -1: '#000000'}  # Light green for 1, light orange for 0, black for -1
    legend_labels = {-1: 'Noise', 0: 'Not Good', 1: 'Good'}

    # Plot the true labels with custom colors and transparency
    color_mapped_true_labels = [colors[label] for label in true_labels]
    axs[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=color_mapped_true_labels, marker='o', alpha=0.7, s=50)
    axs[0, 0].set_title('True Labels')
    
    # Create custom legend for the true labels plot
    handles = [mpatches.Patch(color=colors[key], label=legend_labels[key]) for key in colors]
    axs[0, 0].legend(handles=handles, loc='best')
    
    # Plot the clustering results
    for idx, (name, labels) in enumerate(list(clustering_results.items())[:3]):  # Limit to 3 clustering results
        row, col = divmod(idx + 1, 2)
        # Map colors based on labels
        color_mapped_labels = [colors[label] for label in labels]
        axs[row, col].scatter(X_pca[:, 0], X_pca[:, 1], c=color_mapped_labels, marker='o', alpha=0.7, s=50)
        axs[row, col].set_title(f'{name} Clustering')
        
        # Create custom legend for each clustering plot
        axs[row, col].legend(handles=handles, loc='best')
    
    plt.tight_layout()
    plt.show()

# Matching cluster labels to true labels using the Hungarian algorithm for optimal matching
def match_labels(true_labels, pred_labels):
    contingency_matrix = confusion_matrix(true_labels, pred_labels)
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    matched_labels = np.zeros_like(pred_labels)
    for i, j in zip(row_ind, col_ind):
        matched_labels[pred_labels == j] = i
    return matched_labels

# plot confusion matrix with metrics as print
def plot_confusion_matrix(true_labels, pred_labels, title):
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    accuracy = np.trace(cm) / np.sum(cm)
    
    # Create annotations for each cell in the confusion matrix
    labels = [[f'{value}\n{percentage:.2%}' for value, percentage in zip(row, row/row.sum())] for row in cm]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=labels, fmt="", cmap='Blues', xticklabels=['Bad products', 'Good products'],
                yticklabels=['Bad products', 'Good products'])
    plt.xlabel('Predicted values', fontsize=16)
    plt.ylabel('Actual values', fontsize=16)
    plt.gca().xaxis.set_ticks_position('top')
    plt.gca().xaxis.set_label_position('top')
    plt.title(f'{title}\nAccuracy: {accuracy:.2f}', fontsize=18, pad=20)
    
    # Save the confusion matrix as an image
    plt.savefig(f'confusion_matrix.png', dpi=300)
    plt.show()

    return accuracy


#%% PRAIPLOT FOR PCA = 5
pca = PCA(n_components=5)
X_pca = pca.fit_transform(df_numeric)
X_pca = pd.DataFrame(X_pca)
X_pca["true_labels"] = y_true
sns.pairplot(X_pca, hue="true_labels")


#%% CLUSTER
clustering_results, best_params = grid_search_clustering(X_scaled)
visualize_clusters(X_scaled, y_true, clustering_results)


#%% Cluster quality evaluation with confusion matrix, accuracy, ARI, and NMI
accuracies = {}
ari_scores = {}
nmi_scores = {}

for name, labels in clustering_results.items():
    matched_labels = match_labels(y_true, labels)
    accuracy = plot_confusion_matrix(y_true, matched_labels, f'{name} Clustering Confusion Matrix')
    accuracies[name] = accuracy
    
    # Calculate ARI and NMI
    ari = adjusted_rand_score(y_true, matched_labels)
    nmi = normalized_mutual_info_score(y_true, matched_labels)
    ari_scores[name] = ari
    nmi_scores[name] = nmi
    
    print(f'Best Parameters: {best_params[name]}')
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Adjusted Rand Index (ARI): {ari:.2f}')
    print(f'Normalized Mutual Information (NMI): {nmi:.2f}')
    print('-' * 30)
