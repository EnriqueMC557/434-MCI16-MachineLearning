#!/usr/bin/env python3
"""Methods to streamline data visualization.
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="darkgrid")

__author__ = "Enrique Mena Camilo"
__email__ = "enriquemece97@gmail.com"



def plot_clustering_result(
        X: np.ndarray, y: np.ndarray, centroids: np.ndarray = None, 
        title: str = None, filename: str = None
    ) -> None:
    """Plot the result of a clustering algorithm.

    :param np.ndarray X: Data to cluster.
    :param np.ndarray y: Cluster labels.
    :param np.ndarray centroids: Cluster centroids. Defaults to None.
    :param str title: Plot title. Defaults to None.
    :param str filename: Filename to save the plot. Defaults to None.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, s=100, ax=ax, palette="tab10")
    
    if centroids is not None:
        sns.scatterplot(
            x=centroids[:, 0], y=centroids[:, 1], 
            color="black", s=100, marker="x", linewidth=2, 
            ax=ax, label="Centroids"
        )
    
    if title is not None:
        ax.set_title(f"{title}")
    
    if filename is not None:
        plt.savefig(f"{filename}.png")

    plt.show()
