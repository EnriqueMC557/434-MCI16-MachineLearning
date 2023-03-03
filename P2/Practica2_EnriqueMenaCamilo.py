#!/usr/bin/env python3
"""Machine learning: Practice 2.
Implement data imputation methods for categorical and continuous attributes.
By: Enrique Mena Camilo.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

sns.set_theme(style="darkgrid")

__author__ = "Enrique Mena Camilo"
__email__ = "enriquemece97@gmail.com"


def predict_linear_regression(x_data: np.ndarray, w_data: np.ndarray) -> np.ndarray:
    """Gets predictions using the linear regression method.

    :param np.ndarray x_data: Data with which the prediction will be made.
    :param np.ndarray w_data: Weights obtained from linear regression training.
    :return np.ndarray: Results obtained from the prediction.
    """
    x_data = np.append(np.ones((len(x_data), 1)), x_data, axis=1)
    return np.dot(x_data, w_data)


def train_linear_regression(x_data: np.ndarray, y_data: np.ndarray, lr: float = 0.1, it: int = 1000) -> np.ndarray:
    """Fit linear regression model.

    :param np.ndarray x_data: Training data.
    :param np.ndarray y_data: Target values.
    :param float lr: Learning rate. Defaults to 0.1.
    :param int it: Iterations. Defaults to 1000.
    :return np.ndarray: Weights from linear regression.
    """
    def cost(x_: np.ndarray, y_: np.ndarray, w_: np.ndarray) -> float:
        """MSE as cost function.

        :param np.ndarray x_: Attributes values.
        :param np.ndarray y_: Target values
        :param np.ndarray w_: Weights values.
        :return float:
        """
        return 1/(2 * len(y_)) * np.dot((np.dot(x_, w_) - y_).T, (np.dot(x_, w_) - y_))

    x_data, y_data = x_data.copy(), y_data.copy()

    x_data = np.append(np.ones((len(x_data), 1)), x_data, axis=1)
    x_len = len(x_data)
    theta = np.ones((x_data.shape[1], 1))

    for _ in tqdm(range(it)):
        theta = theta - lr * (1/x_len) * np.dot(x_data.T, np.dot(x_data, theta) - y_data)
    print(f"Final cost: {cost(x_data, y_data, theta)}")

    return theta


def target_mean_imputation(dataset: pd.DataFrame, feature: str, target: str, feature_type: str = "quanti") -> pd.DataFrame:
    """Performs data imputation using the target mean method.

    :param pd.DataFrame dataset: Full dataset.
    :param str feature: Feature name to imputate.
    :param str target: Target name.
    :param str feature_type: One of ['quanti', 'quali'].
    :return pd.DataFrame: Feature with imputated data.
    """
    to_fill = dataset[feature].copy()

    for class_ in dataset[target].unique():
        if feature_type == "quali":
            estimated = to_fill.loc[dataset[target] == class_].mode()[0]
        elif feature_type == "quanti":
            estimated = to_fill.loc[dataset[target] == class_].mean()
        else:
            raise ValueError("'feature_type' must be one of ['quanti', 'quali']")

        to_fill.loc[(dataset[target] == class_) & (to_fill.isna())] = estimated

    return to_fill


def random_imputation(dataset: pd.DataFrame, feature: str, feature_type: str = "quanti") -> pd.DataFrame:
    """Performs data imputation using the random method.

    :param pd.Dataframe dataset: Full dataset.
    :param str feature: Feature name to imputate.
    :param str feature_type: One of ['quanti', 'quali'].
    :return pd.DataFrame: Feature with imputated data.
    """
    to_fill = dataset[feature].copy()
    nan_count = to_fill.isna().sum()

    if feature_type == "quanti":
        estimated = np.random.uniform(low=to_fill.min(), high=to_fill.max()+1, size=nan_count)
    elif feature_type == "quali":
        estimated = np.random.randint(low=to_fill.min(), high=to_fill.max()+1, size=nan_count)
    else:
        raise ValueError("'feature_type' must be one of ['quanti', 'quali']")

    estimated = pd.Series(estimated, index=to_fill.loc[to_fill.isna()].index.to_list())
    return to_fill.fillna(estimated)


class DatasetNanHandler:
    def __init__(self, dataset: pd.DataFrame, target: str):
        self.dataset = dataset.copy()
        self.columns = dataset.columns.to_list()
        self.target = target
        self.features = dataset.drop(columns=[target]).columns.to_list()

        self.nan_count = None

    def get_nan_count(self, rebuild: bool = False) -> pd.DataFrame:
        """Build information about missing values in the data set.

        :param bool rebuild: Flag to indicate whether to use previous data or recalculate. Defaults to False.
        :return pd.DataFrame: DataFrame with information about nan values around full dataset.
        """
        if self.nan_count is None or rebuild:
            nan_count = list()
            for column in self.columns:
                nan_count.append({
                    "column": column,
                    "total_count": self.dataset[column].shape[0],
                    "non_nan_count": self.dataset[column].count(),
                    "nan_count": self.dataset[column].isna().sum(),
                    "nan_percentage": round(self.dataset[column].isna().sum()/self.dataset[column].count(), 4)
                })

            self.nan_count = pd.DataFrame(nan_count).set_index("column")
        return self.nan_count

    def get_histograms(self, shape: tuple, sufix: str = None):
        """Build column's histograms in subplots matrix.

        :param tuple shape: Subplots shape.
        :param str sufix: Sufix in the png file. Defaults to None.
        """
        fig, axes = plt.subplots(shape[0], shape[1])
        fig.suptitle(f"Dataset's columns histograms {sufix}")
        fig.set_size_inches(10, 10)
        sufix = f"_{sufix}" if sufix else ""

        for ax, column in zip(axes.flatten(), self.columns):
            sns.histplot(data=self.dataset, x=column, ax=ax)
            ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"./figures/dataset_histogram{sufix}.png", bbox_inches='tight')
        plt.show()

    def get_heatmap(self):
        """Build dataset correlation heatmap.
        """
        fig, ax = plt.subplots(1)
        fig.suptitle(f"Dataset's heatmap")
        fig.set_size_inches(10, 10)
        sns.heatmap(data=self.dataset.corr(), annot=True, cmap='BrBG', vmin=-1, vmax=1)
        plt.savefig(f"./figures/dataset_heatmap.png", bbox_inches='tight')
        plt.show()

    def fill_nan(self, method: str, feature: str, **kwargs):
        """Perform data imputation according to the choiced method.

        :param str method: Once of ['mean', 'mode', 'target_mean', 'random', 'regression']
        :param str feature: Feature name to imputate.
        :param kwargs:
        :return:
        """
        if method == "mean":
            estimated = self.dataset[feature].mean()
            self.dataset[feature].fillna(estimated, inplace=True)
            return

        if method == "mode":
            estimated = self.dataset[feature].mode()[0]
            self.dataset[feature].fillna(estimated, inplace=True)
            return

        if method == "target_mean":
            estimated = target_mean_imputation(self.dataset, feature, self.target, kwargs.get("feature_type", "quali"))
            self.dataset[feature] = estimated
            return

        if method == "random":
            estimated = random_imputation(self.dataset, feature, kwargs.get("feature_type", "quali"))
            self.dataset[feature] = estimated
            return

        if method == "regression":
            if "predictors" in kwargs:
                predictors = kwargs["predictors"]
            else:
                raise TypeError("Missing 1 required keyword-only argument: 'predictors'")

            # Prepare predictors and target
            x = self.dataset[self.dataset[feature].notnull()][predictors].to_numpy()
            y = self.dataset[self.dataset[feature].notnull()][feature].to_numpy()
            y = y.reshape(y.shape + (1,))

            # Normalize predictors
            scaler = MinMaxScaler()
            scaler.fit(x)
            x = scaler.transform(x)

            # Train regression
            weights = train_linear_regression(x, y)

            # Fill NaN values
            x_nan = self.dataset[self.dataset[feature].isnull()][predictors]
            to_fill_index = x_nan.index.to_list()
            x_nan = scaler.transform(x_nan.to_numpy())
            estimated = predict_linear_regression(x_nan, weights)[:, 0]
            self.dataset.loc[to_fill_index, feature] = estimated
            return

        raise ValueError("Invalid 'method'")


if __name__ == "__main__":
    data = pd.read_csv("../data/stroke_nan.csv")
    data_handler = DatasetNanHandler(data, target="stroke")

    print("====NaN values count (original)====")
    print(data_handler.get_nan_count())

    print("====Histograms (original)====")
    data_handler.get_histograms((4, 3), sufix="original")

    print("====Heatmap====")
    data_handler.get_heatmap()

    print("====Mode imputation===")
    data_handler.fill_nan("mode", "sex")

    print("====Target mean imputation====")
    data_handler.fill_nan("target_mean", "hypertension", feature_type="quali")

    print("====Random imputation====")
    data_handler.fill_nan("random", "Residence_type", feature_type="quali")
    data_handler.fill_nan("random", "work_type", feature_type="quali")

    print("====Regression imputation====")
    data_handler.fill_nan("regression", "bmi", predictors=["avg_glucose_level"])

    # bmi vs glucose dispersion
    ax = sns.pairplot(data, y_vars=["bmi"], x_vars=["avg_glucose_level"])
    ax.fig.set_size_inches(8, 8)
    plt.savefig("./figures/bmi_glucose.png", bbox_inches='tight')
    plt.show()

    print("====NaN values count (after imputation)====")
    print(data_handler.get_nan_count(rebuild=True))

    print("====Histograms (imputated)====")
    data_handler.get_histograms((4, 3), sufix="imputated")
