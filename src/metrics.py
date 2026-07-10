import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, accuracy_score,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_mutual_info_score, adjusted_rand_score, homogeneity_score, completeness_score
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def sample_classification_score(
    test_value: float,
    test_cluster: int,
    train_distribution: list,
    criterion: str
) -> float:
    """
    Calculate test sample statistics with respect to train_distribution.

    Args:
        test_value (float): Value of the sample.
        test_cluster (int): Cluster of the sample.
        train_distribution (list): List of train values distributions across clusters [num_clusters, num_samples].
        criterion (str): A way to calculate statistics of test sample. Could be 'left-sided' or 'right-sided'.

    Returns:
        float: Test sample statistics.
    """
    cluster_size = len(train_distribution[test_cluster])

    if cluster_size == 0:
        return 0.0 if criterion == 'left-sided' else 1.0

    if criterion == 'left-sided':
        return 1.0 - sum(value < test_value for value in train_distribution[test_cluster]) / cluster_size
    elif criterion == 'right-sided':
        return sum(value < test_value for value in train_distribution[test_cluster]) / cluster_size
    else:
        raise ValueError("Invalid criterion. Must be 'left-sided' or 'right-sided'.")


def classification_scores(
    train_features: torch.Tensor, 
    test_features: torch.Tensor, 
    features_type: str = 'clusters_probas', 
    criterion: str = 'left-sided', 
    clusters_num: int = None,
    cluster_centers: np.ndarray = None,
    train_clusters: np.ndarray = None

) -> np.ndarray:
    """
    Calculate test samples statistics with respect to train_distribution.

    Args:
        train_features (torch.Tensor): Train values.
        test_features (torch.Tensor): Test values.
        features_type (str): If 'clusters_probas', values are probabilities of samples belonging to clusters.
                             If 'features', values are abstract features.
        criterion (str): A way to calculate statistics of test sample. Could be 'left-sided' or 'right-sided'.
        clusters_num (int | None): If features_type = 'features', the number of clusters for data clustering.
        
    Returns:
        np.ndarray: Test samples statistics.
    """

    if torch.is_tensor(train_features):
        train_features = train_features.numpy()

    if torch.is_tensor(test_features):
        test_features = test_features.numpy()

    if features_type == 'clusters_probas':

        train_clusters = train_features.argmax(axis=-1)
        train_distribution = [
            train_features[train_clusters == i, i] for i in range(train_features.shape[1])
        ]
        test_clusters = test_features.argmax(axis=-1)
        test_values = test_features.max(axis=-1)

    elif features_type == 'features':
        if clusters_num is None:
            raise ValueError("clusters_num must be provided when features_type is 'features'")

        train_distribution = [
            np.sqrt(
                np.sum(
                    (train_features[train_clusters == i] - cluster_centers[i][np.newaxis, :]) ** 2, axis=-1
                )
            ).reshape(-1)
            for i in range(clusters_num)
        ]

        dists = np.sqrt(
            np.sum(
                (test_features[:, np.newaxis, :] - cluster_centers[np.newaxis, :, :]) ** 2, axis=-1
            )
        )
        test_clusters = dists.argmin(axis=-1)
        test_values = dists.min(axis=-1)

    else:
        raise ValueError("Invalid features_type. Must be 'clusters_probas' or 'features'.")

    scores = [
        sample_classification_score(test_values[i], test_clusters[i], train_distribution, criterion) 
        for i in range(test_features.shape[0])
    ]

    return np.array(scores)

def classification_metrics(scores: np.ndarray, targets: np.ndarray, threshold: float, prefix: str) -> dict:
    """
    Calculate evaluation metrics.

    Args:
        scores (np.ndarray): Predicted scores.
        targets (np.ndarray): True labels.
        threshold (float): Threshold for binary classification.
        prefix (str): Prefix for metric names in the output dictionary.

    Returns:
        dict: Dictionary containing evaluation metrics with the given prefix.
    """
    predictions = (scores > threshold).astype(int)

    return {
        f"{prefix}_roc_auc": roc_auc_score(targets, scores),
        f"{prefix}_f1": f1_score(targets, predictions),
        f"{prefix}_precision": precision_score(targets, predictions),
        f"{prefix}_recall": recall_score(targets, predictions),
        f"{prefix}_accuracy": accuracy_score(targets, predictions),
    }


def process_features(train_features, test_features, clusters_num) -> dict:
    """
    Perform clustering and dimensionality reduction on features.

    Args:
        train_features (np.ndarray): Training features.
        test_features (np.ndarray): Testing features.
        clusters_num (int): Number of clusters.

    Returns:
        tuple: Transformed train/test features, cluster assignments, and cluster centers.
    """
    k_means = KMeans(n_clusters=clusters_num)
    pca = PCA(n_components=3)

    train_features = pca.fit_transform(train_features)
    test_features = pca.transform(test_features)

    train_clusters = k_means.fit_predict(train_features)
    test_clusters = k_means.predict(test_features)
    cluster_centers = k_means.cluster_centers_

    return train_features, test_features, train_clusters, test_clusters, cluster_centers


def clustering_metrics(train_features, test_features, clusters_num, prefix) -> dict:
    """
    Calculate clustering metrics for train and test features.

    Args:
        train_features (np.ndarray): Training features.
        test_features (np.ndarray): Testing features.
        clusters_num (int): Number of clusters.
        prefix (str): Prefix for metric names in the output dictionary.

    Returns:
        dict: Dictionary containing clustering metrics.
    """
    train_features, test_features, train_clusters, test_clusters, _ = process_features(
        train_features, test_features, clusters_num
    )

    return {
        f"{prefix}_silhouette_train": silhouette_score(train_features, train_clusters),
        f"{prefix}_silhouette_test": silhouette_score(test_features, test_clusters),
        f"{prefix}_davies_bouldin_train": davies_bouldin_score(train_features, train_clusters),
        f"{prefix}_davies_bouldin_test": davies_bouldin_score(test_features, test_clusters),
        f"{prefix}_calinski_harabasz_train": calinski_harabasz_score(train_features, train_clusters),
        f"{prefix}_calinski_harabasz_test": calinski_harabasz_score(test_features, test_clusters),
    }


def get_supervised_metrics_features(
    train_features, test_features, targets, clusters_numbers, test_devices, train_devices
) -> dict:
    """
    Calculate supervised metrics for feature-based clustering.

    Args:
        train_features (np.ndarray): Training features.
        test_features (np.ndarray): Testing features.
        targets (np.ndarray): True labels.
        clusters_numbers (list): List of cluster numbers to evaluate.
        test_devices (np.ndarray): True cluster assignments for test data.
        train_devices (np.ndarray): True cluster assignments for train data.

    Returns:
        dict: Dictionary containing supervised metrics.
    """
    all_metrics = {}

    for clusters_number in clusters_numbers:
        train_features, test_features, train_clusters,\
        test_clusters, cluster_centers = process_features(train_features, test_features, clusters_number)

        clusters_super_metrics = {
            f"{clusters_number}_nmi": adjusted_mutual_info_score(test_devices, test_clusters),
            f"{clusters_number}_ari": adjusted_rand_score(test_devices, test_clusters),
            f"{clusters_number}_homogeneity": homogeneity_score(test_devices, test_clusters),
            f"{clusters_number}_completeness_score": completeness_score(test_devices, test_clusters),
        }

        scores = classification_scores(
            train_features,
            test_features,
            criterion="right-sided",
            features_type="features",
            clusters_num=clusters_number,
            cluster_centers = cluster_centers,
            train_clusters = train_clusters
        )

        all_metrics |= classification_metrics(scores, targets, 0.95, clusters_number) | clusters_super_metrics

    return all_metrics


def get_supervised_metrics_probas(train_features, test_features, targets, clusters_numbers = None, threshold=0.95) -> dict:
    """
    Calculate supervised metrics for probability-based clustering.

    Args:
        train_features (np.ndarray): Training features.
        test_features (np.ndarray): Testing features.
        targets (np.ndarray): True labels.
        clusters_numbers (list): List of cluster numbers to evaluate. Not Used

    Returns:
        dict: Dictionary containing supervised metrics.
    """
    scores = classification_scores(
        train_features,
        test_features,
        criterion="left-sided",
        features_type="clusters_probas",
    )

    return classification_metrics(scores, targets, threshold, "probas")


def get_unsupervised_metrics_features(train_features, test_features, clusters_numbers) -> dict:
    """
    Calculate unsupervised metrics for feature-based clustering.

    Args:
        train_features (np.ndarray): Training features.
        test_features (np.ndarray): Testing features.
        clusters_numbers (list): List of cluster numbers to evaluate.

    Returns:
        dict: Dictionary containing unsupervised metrics.
    """
    all_metrics = {}

    for clusters_num in clusters_numbers:
        all_metrics |= clustering_metrics(train_features, test_features, clusters_num, clusters_num)

    return all_metrics