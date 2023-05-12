#!/usr/bin/env python3
"""Contains methods that allow expediting tasks of exploratory data analysis (EDA).
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

__author__ = "Enrique Mena Camilo"
__email__ = "enriquemece97@gmail.com"

sns.set_theme(style="darkgrid")


def get_nan_count(data: pd.DataFrame) -> pd.DataFrame:
    """Get the number and percentage of NaN values per column.

    :param pd.DataFrame data: Dataframe to analyze.
    :return pd.DataFrame: Dataframe with the number and percentage of NaN values per column.
    """
    nan_count = data.isna().sum()
    nan_count = nan_count.to_frame()
    nan_count.columns = ["nan_count"]
    nan_count["nan_percentage"] = round(nan_count["nan_count"] / data.shape[0], 4) * 100
    nan_count = nan_count.sort_values(by="nan_count", ascending=False)
    return nan_count


def code_categorical(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """Code the categorical variables of a column.

    :param pd.DataFrame data: Dataframe to analyze.
    :param str column: Column to code.
    :return pd.DataFrame: Dataframe with the coded column.
    """
    data[column] = data[column].astype("category")
    data[column] = data[column].cat.codes
    return data


def plot_distribution(data: pd.DataFrame, column: str, **kwargs) -> None:
    """Plot the distribution of a column.

    :param pd.DataFrame data: Dataframe to analyze.
    :param str column: Column to analyze.
    """
    figures_path = kwargs.get("figures_path", "./")
    sufix = kwargs.get("sufix", None)
    sufix = f"_{sufix}" if sufix else ""

    fig, ax = plt.subplots(1, 2, figsize=(18, 4))
    sns.histplot(x=column, data=data, ax=ax[0], kde=True)
    sns.boxplot(x=column, data=data, ax=ax[1])
    fig.suptitle(f"{column}{sufix}")
    plt.savefig(f"{figures_path}{column}_distribution{sufix}.png")
    plt.show()


def plot_count(data: pd.DataFrame, column: str, **kwargs) -> None:
    """Plot the count of each class.

    :param pd.DataFrame data: Dataframe to analyze.
    :param str column: Column to analyze.
    """
    figures_path = kwargs.get("figures_path", "./")
    sufix = kwargs.get("sufix", None)
    sufix = f"_{sufix}" if sufix else ""

    fig, ax = plt.subplots(1, 1, figsize=(18, 4))
    sns.countplot(x=column, data=data, ax=ax, order=data[column].value_counts().index)
    fig.suptitle(f"Balance of {column}{sufix}")
    abs_values = data[column].value_counts(ascending=False)
    rel_values = data[column].value_counts(ascending=False, normalize=True).values * 100
    lbls = [f'{p[0]} ({p[1]:.0f}%)' for p in zip(abs_values, rel_values)]
    ax.bar_label(container=ax.containers[0], labels=lbls)
    plt.savefig(f"{figures_path}{column}_balance{sufix}.png")
    plt.show()
