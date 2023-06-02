#!/usr/bin/env python3
"""Contains methods that allow you to reduce the dimension of the data.
"""

import numpy as np
import pandas as pd

__author__ = "Enrique Mena Camilo"
__email__ = "enriquemece97@gmail.com"


def pca(X: pd.DataFrame, n_components: int, normalize_data: bool = True) -> pd.DataFrame:
    """
    Perform principal component analysis (PCA) on the input data.

    :param pd.DataFrame X: Input data.
    :param int n_components: Number of principal components to keep.
    :param bool normalize_data: Whether to normalize the data before performing PCA.
    :return pd.DataFrame: Transformed data with principal components as columns.
    """
    X_ = X.values

    if normalize_data:
        X_ = (X_ - X_.mean()) / X_.std()

    # Calculate the covariance matrix
    cov_mat = np.cov(X_.T)

    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)

    # Sort the eigenvalues in descending order and select the top n_components
    sorted_indices = np.argsort(eig_vals)[::-1]
    sorted_eigen_vectors = eig_vecs[:, sorted_indices]

    # Create the projection matrix and project the data onto the new subspace
    projection_matrix = sorted_eigen_vectors[:, :n_components]
    X_pca = X_.dot(projection_matrix)

    # Create a new DataFrame with the principal components
    columns = ['PC' + str(i+1) for i in range(n_components)]
    X_pca_df = pd.DataFrame(X_pca, columns=columns)

    return X_pca_df


def pca_features(X: pd.DataFrame, normalize_data: bool = True) -> pd.DataFrame:
    """
    Gets the relevance of each attribute using the PCA method.

    :param pd.DataFrame X: Input data.
    :param bool normalize_data: Whether to normalize the data before performing PCA.
    :return pd.DataFrame: Principal components as columns.
    """
    X_ = X.values

    if normalize_data:
        X_ = (X_ - X_.mean()) / X_.std()

    # Calculate the covariance matrix
    cov_mat = np.cov(X_.T)

    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    eig_vals, _ = np.linalg.eig(cov_mat)

    # Get the percentage of covariance of each feature
    relevance = (eig_vals / eig_vals.sum()) * 100
    relevance = pd.DataFrame(relevance, index=X.columns, columns=["relevance"])
    relevance = relevance.sort_values(by="relevance", ascending=False)
    
    return relevance
    
