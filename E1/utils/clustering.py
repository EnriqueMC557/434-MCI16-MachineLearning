#!/usr/bin/env python3
"""Contains methods that implement clustering algorithms.
"""

import numpy as np

from sklearn.cluster import AffinityPropagation

from tqdm import tqdm

__author__ = "Enrique Mena Camilo"
__email__ = "enriquemece97@gmail.com"


def k_means(X: np.ndarray, k: int, max_iter: int = 100, tol: float = 1e-3) -> tuple:
    """Implementation of the k-means algorithm.

    :param np.ndarray X: Data to cluster.
    :param int k: Number of clusters.
    :param int max_iter: Maximum number of iterations. Defaults to 1000.
    :param float tol: Tolerance. Defaults to 1e-3.
    :return np.ndarray: Cluster labels.
    :return np.ndarray: Centroids.
    """
    # Initialize centroids
    centroids = np.random.rand(k, X.shape[1])
    
    # Initialize cluster labels
    labels = np.zeros(X.shape[0])
    
    # Initialize error
    error = np.inf
    
    # Iterate until convergence or max_iter
    for _ in tqdm(range(max_iter)):
        # Compute distances and assign labels
        distances = np.linalg.norm(X[:, np.newaxis, :] - centroids, axis=2)
        new_labels = np.argmin(distances, axis=1)
        
        # Compute new centroids
        new_centroids = np.array([X[new_labels == j, :].mean(axis=0) for j in range(k)])
        
        # Check for convergence
        error = np.linalg.norm(new_centroids - centroids)
        if error < tol:
            break
        
        # Update centroids and labels
        centroids = new_centroids
        labels = new_labels
    
    return labels, centroids


def affinity_propagation(X: np.ndarray, damping: float = 0.5, max_iter: int = 100, conv_iter: int = 10) -> tuple:
    """Run the Affinity Propagation algorithm on the input data.

    :param np.ndarray X: Data to cluster.
    :param float damping: Damping factor. Defaults to 0.5.
    :param int max_iter: Maximum number of iterations. Defaults to 100.
    :param int conv_iter: Number of iterations to consider convergence. Defaults to 10.
    :return np.ndarray: Cluster centers.
    :return np.ndarray: Cluster labels.
    """
    af = AffinityPropagation(damping=damping, max_iter=max_iter, convergence_iter=conv_iter)
    af.fit(X)

    return af.labels_, af.cluster_centers_
