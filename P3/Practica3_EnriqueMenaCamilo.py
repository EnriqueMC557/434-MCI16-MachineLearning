#!/usr/bin/env python3
"""Machine learning: Practice 3.
Implement normalization methods and class balancing.
By: Enrique Mena Camilo.
"""

import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Union, Tuple

from collections import Counter

sns.set_theme(style="darkgrid")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

__author__ = "Enrique Mena Camilo"
__email__ = "enriquemece97@gmail.com"


def get_histogram(feature: Union[np.ndarray, pd.DataFrame, pd.Series], title: str):
    """_summary_

    :param Union[np.ndarray, pd.DataFrame, pd.Series] feature:
    :param str title:
    """
    fig, ax = plt.subplots(1, 1)
    fig.suptitle(f"Histogram {title}")
    fig.set_size_inches(10, 10)

    sns.histplot(feature, ax=ax, kde=True)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f"./figures/histogram_{'_'.join(title.lower().split())}.png", bbox_inches="tight")
    plt.show()


def get_classes_distribution(dataset: pd.DataFrame, target_name: str, title: str):
    """

    :param dataset:
    :param target_name:
    :param title:
    :return:
    """
    plt.figure(figsize=(10, 10))
    ax = sns.countplot(data=dataset, x=target_name,
                       order=dataset[target_name].value_counts(ascending=False).index)
    abs_values = dataset[target_name].value_counts(ascending=False)
    rel_values = dataset[target_name].value_counts(ascending=False, normalize=True).values * 100
    lbls = [f"{p[0]} ({p[1]:.0f}%)" for p in zip(abs_values, rel_values)]
    _ = ax.bar_label(container=ax.containers[0], labels=lbls)
    _ = ax.set_title(f"Classes distribution ({title})", fontsize=12)
    plt.savefig(f"./figures/classes_distribution_{title.lower()}.png", bbox_inches="tight")
    plt.show()


def min_max_normalization(feature: Union[np.ndarray, pd.DataFrame, pd.Series], desired_range: tuple = (0, 1)) -> np.ndarray:
    """_summary_

    :param Union[np.ndarray, pd.DataFrame, pd.Series] feature: _description_
    :param tuple desired_range: _description_, defaults to (0, 1)
    :return np.ndarray: _description_
    """
    feature_ = feature.copy()
    if type(feature) in [pd.DataFrame, pd.Series]:
        feature_ = feature.to_numpy()

    feature_ = (feature_ - feature_.min(axis=0)) / (feature_.max(axis=0) - feature_.min(axis=0))
    return feature_ * (desired_range[1] - desired_range[0]) + desired_range[0]


def z_score_normalization(feature: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
    """_summary_

    :param Union[np.ndarray, pd.DataFrame, pd.Series] feature: _description_
    :return np.ndarray: _description_
    """
    feature_ = feature.copy()
    if type(feature) in [pd.DataFrame, pd.Series]:
        feature_ = feature_.to_numpy()

    return (feature_ - feature_.mean(axis=0)) / feature_.std(axis=0)


def l1_normalization(feature: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
    """_summary_

    :param Union[np.ndarray, pd.DataFrame, pd.Series] feature: _description_
    :return np.ndarray: _description_
    """
    feature_ = feature.copy()
    if type(feature) in [pd.DataFrame, pd.Series]:
        feature_ = feature_.to_numpy()

    norm = np.sum(np.abs(feature_))
    return feature_ / norm


def roulette_class_balancing(dataset: pd.DataFrame, N: int, target_name: str, minor_class: int) -> pd.DataFrame:
    """Generate synthetic data from roulette method.

    :param pd.DataFrame dataset: Dataset to which synthetic data will be added
    :param int N: Number of synthetic instances to generate.
    :param str target_name: Column name of target variable.
    :param int minor_class: Label of the class with the least amount of data.
    :return pd.DataFrame: Balanced dataset
    """
    ds = []
    data_minor = dataset.copy()
    data_minor = data_minor[data_minor[target_name] == minor_class]
    columns = data_minor.columns.to_list()
    for j in range(N):
        row = []
        for i in range(len(columns)):
            # Unique value
            item_counts_key = data_minor[columns[i]].value_counts().index.tolist()

            # Probability that this value appears in the database
            item_counts = data_minor[columns[i]].value_counts(normalize=True)

            # Cumulative sum of probability to prioritize values
            cumsum_p = item_counts.cumsum()
            w = cumsum_p.to_numpy()

            # Generate a random percentage
            r = random.random()

            # Find the closest percentage achieved by unique values
            magic = np.argwhere(r <= w)

            # Get the index of each value
            x = int(magic[0])

            # Get the value to generate synthetic data
            dato = item_counts_key[x]
            row.append(dato)

        # Add the synthetically created value to the dataset
        ds.append(row)

    # Build new dataset
    d_syn = pd.DataFrame(ds, columns=columns)
    return pd.concat([dataset, d_syn], ignore_index=True)


def smote_class_balancing(
        features: Union[np.ndarray, pd.DataFrame, pd.Series],
        target: Union[np.ndarray, pd.DataFrame, pd.Series],
        minor_class: int, k: int, N: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates new synthetic instances of the minority class using the SMOTE algorithm.

    :param Union[np.ndarray, pd.DataFrame, pd.Series] features: Features array with shape (n_samples, n_features).
    :param Union[np.ndarray, pd.DataFrame, pd.Series] target: Target array with shape (n_samples,).
    :param int minor_class: Label of the class with the least amount of data.
    :param int k: Number of nearest neighbors to consider to generate new synthetic instances.
    :param int N: Number of synthetic instances to generate.
    :return Tuple[np.ndarray, np.ndarray]: (Balanced features array, balanced target array).
    """
    features_ = features.copy()
    if type(features) in [pd.DataFrame, pd.Series]:
        features_ = features_.to_numpy()

    # Splits majority and minority class instances
    minority_class = features_[target == minor_class]
    majority_class = features_[target != minor_class]

    # Initializes the feature array and label vector for the new synthetic instances
    synthetic_features = np.zeros((N, minority_class.shape[1]))
    synthetic_labels = np.ones(N)

    # SMOTE algorithm
    for i in range(N):
        # Select a random instance of the minority class
        j = np.random.choice(minority_class.shape[0])

        # Find the k nearest neighbors of the selected instance in the feature space
        knn_indices = np.argsort(np.sum((minority_class - minority_class[j])**2, axis=1))[:k]

        # Randomly select one of the k nearest neighbors
        m = np.random.choice(knn_indices)

        # Generates a new synthetic instance by interpolating between the selected instance and the selected neighbor
        synthetic_features[i, :] = minority_class[j, :] + np.random.rand() * (minority_class[m, :] - minority_class[j, :])

    # Concatenates the new synthetic instances with the instances of the original minority class
    x_resampled = np.vstack((minority_class, synthetic_features))
    y_resampled = np.concatenate((np.zeros(minority_class.shape[0]), np.zeros(N)))

    # Concatenates the instances of the original majority class
    x_resampled = np.vstack((x_resampled, majority_class))
    y_resampled = np.concatenate((y_resampled, np.ones(majority_class.shape[0])))

    return x_resampled, y_resampled


if __name__ == "__main__":
    data = pd.read_csv("../data/cbds.csv")

    # ==== DATA PREPROCESSING ====
    # Does the dataset have missing data?
    print(data.info())
    # R: No, the dataset is complete

    # Get histograms with unnormalized data
    columns = data.columns.to_list()
    for column in columns:
        get_histogram(data[column], f"{column} unnormalized")

    # ==== DATA NORMALIZATION ====
    data_norm = data.copy()
    # Min-Max normalizations
    features = ["age_at_ercp", "race", "parity", "stones_on_bd", "intraductal_filling", "cystic_duct_filling"]
    for feature in features:
        # The selected range is made in this way because there are no negative observations.
        data_norm[feature] = min_max_normalization(data_norm[[feature]], desired_range=(0, 1))

    # L1 normalizations
    features = ["pyobilia_ercp", "stone_shape_ercp", "stone_color_ercp"]
    for feature in features:
        data_norm[feature] = l1_normalization((data_norm[[feature]]))

    # Z-score normalizations
    features = ["bmi", "peak_bili", "cbd_diameter_us", "cbd_diameter_mrcp", "cbd_diameter_ercp"]
    for feature in features:
        data_norm[feature] = z_score_normalization(data_norm[[feature]])

    # Get histograms with normalized data
    columns = data.columns.to_list()
    for column in columns:
        get_histogram(data_norm[column], f"{column} normalized")

    # ==== CLASS BALANCING ====
    target = "stone_sludge_ercp"

    # What do we know about the classes? are they balanced?
    get_classes_distribution(data, target, "original")
    # R: We don't have a class balance, we have a ratio close to 85-15

    print("Original standard deviation")
    print(data.std())

    # Roulette class balancing
    data_roulette = roulette_class_balancing(data, 200, target, 0)
    get_classes_distribution(data_roulette, target, "roulette")

    # Get histograms with roulette data
    columns = data_roulette.columns.to_list()
    for column in columns:
        get_histogram(data_roulette[column], f"{column} roulette")

    print("Roulette standard deviation")
    print(data_roulette.std())

    # SMOTE class balancing
    x_balanced, y_balanced = smote_class_balancing(data.drop(columns=[target]), data[target], minor_class=0, k=3, N=200)
    features = data.drop(columns=target).columns.to_list()
    data_smote = pd.DataFrame(data=x_balanced, columns=features)
    data_smote[target] = y_balanced
    get_classes_distribution(data_smote, target, "SMOTE")

    # Get histograms with SMOTE data
    columns = data_smote.columns.to_list()
    for column in columns:
        get_histogram(data_smote[column], f"{column} SMOTE")

    print("SMOTE standard deviation")
    print(data_smote.std())
