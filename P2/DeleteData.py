#!/usr/bin/env python3
"""Script to delete data from data set.
"""

import numpy as np
import pandas as pd

from typing import Union

__author__ = "Enrique Mena Camilo"
__email__ = "enriquemece97@gmail.com"


def delete_data(dataset: pd.DataFrame, percentage: float = 0.15, to_delete: Union[None, list] = None):
    """Performs random data deletion.

    :param dataset: Data set to perform data deletion.
    :param percentage: Percentage of deletion
    :param to_delete: Columns to delete.
    :return pd.DataFrame: Data set with deleted data.
    """
    if percentage <= 0 or percentage >= 1:
        raise ValueError("'percentage' must be more than 0 and less than 1")

    dataset = dataset.copy()
    total_samples = dataset.shape[0]
    to_delete_samples = int(total_samples*percentage)

    if to_delete is None:
        to_delete = dataset.columns.to_list()

    for feature in to_delete:
        delete_samples = np.random.randint(0, total_samples, size=to_delete_samples)
        dataset.loc[delete_samples, feature] = np.nan

    return dataset


if __name__ == "__main__":
    data = pd.read_csv("../data/stroke.csv")
    features_to_delete = data.drop(columns=["sex", "age", "stroke", "heart_disease", "avg_glucose_level",
                                            "ever_married", "smoking_status"]).columns.to_list()
    data_deleted = delete_data(data, to_delete=features_to_delete)
    data_deleted.to_csv("../data/stroke_nan.csv", index=False)
