#!/usr/bin/env python3
"""Machine learning: Practice 4.
Continuous data metrics and regression.
By: Enrique Mena Camilo.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures

sns.set_theme(style="darkgrid")
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

__author__ = "Enrique Mena Camilo"
__email__ = "enriquemece97@gmail.com"


def currency_str_to_float(value: str) -> float:
    """Converts a currency formatted string to a float value.

    :param str value: String to convert, e.g. '$135.04' or '$946,542.094'
    :return float: String converted to float, e.g. 135.09 or 946542.094
    """
    return float(value.replace("$", "").replace(",", "") if isinstance(value, str) else value)


def get_histplot(data: pd.Series, title: str):
    """Generates the histogram of the requested data.

    :param pd.Series data: Data to analyze
    :param str title: Figure title
    """
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(8, 4)
    sns.histplot(data, ax=ax, kde=True)
    ax.set_title(f"'{title}' distribution")
    plt.savefig(f"./figures/histplot_{title.lower()}.png", bbox_inches="tight")
    fig.show()


def get_boxplot(data: pd.Series, title: str):
    """Generates the boxplot of the requested data.

    :param pd.Series data: Data to analyze
    :param str title: Figure title
    """
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(8, 4)
    sns.boxplot(data, ax=ax)
    ax.set_title(f"'{title}' boxplot")
    plt.savefig(f"./figures/boxplot_{title.lower()}.png", bbox_inches="tight")
    fig.show()


def get_distribution_plots(data: pd.Series, title: str):
    """Generates the histogram and boxplot of the requested data.

    :param pd.Series data: Data to analyze
    :param str title: Figure title
    """
    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(18, 4)
    sns.histplot(data, ax=axes[0], kde=True)
    axes[0].set_title(f"'{title}' distribution")
    sns.boxplot(data, ax=axes[1])
    axes[1].set_title(f"'{title}' boxplot")
    axes[1].set_xlabel(data.name)
    plt.savefig(f"./figures/distribution_{title.lower()}.png", bbox_inches="tight")
    fig.show()


def plot_predictions(train_set: np.ndarray, test_set: np.ndarray, train_pred: np.ndarray, test_pred: np.ndarray, title: str):
    """Plots the results of the predictions and the actual data.

    :param np.ndarray train_set: Actual train data
    :param np.ndarray test_set: Actual test data
    :param np.ndarray train_pred: Prediected train data
    :param np.ndarray test_pred: Predictes test data
    :param str title: Figure title
    """
    fig = plt.figure(figsize=(8, 4))
    plt.scatter(train_set[0], train_set[1], s=10, label="Train data")
    plt.scatter(test_set[0], test_set[1], s=10, label="Test data")
    plt.plot(train_set[0], train_pred, linestyle="solid", label="Train predictions")
    plt.plot(test_set[0], test_pred, linestyle="solid", label="Test predictions")
    plt.title(f"{title} results")
    plt.xlabel("Date")
    plt.ylabel("Close/Last")
    plt.legend()
    plt.savefig(f"./figures/{'_'.join(title.lower().split())}_results", bbox_inches="tight")
    fig.show()


def get_heatmap(data: pd.DataFrame):
    """Build dataset correlation heatmap.

    :param pd.DataFrame data: Data to analyze
    """
    fig, ax = plt.subplots(1)
    fig.suptitle(f"Dataset's heatmap")
    fig.set_size_inches(8, 8)
    sns.heatmap(data=data.corr(), annot=True, cmap='BrBG', vmin=-1, vmax=1, ax=ax)
    plt.savefig(f"./figures/dataset_heatmap.png", bbox_inches='tight')
    fig.show()


def MSE(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Squared Error (MSE).

    :param np.ndarray actual: Actual data
    :param np.ndarray predicted: Predicted data
    """
    return np.sum((actual - predicted)**2) / len(actual)


def RMSE(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error (RMSE).

    :param np.ndarray actual: Actual data
    :param np.ndarray predicted: Predicted data
    """
    return np.sqrt(np.sum((actual - predicted)**2) / len(actual))


def MAE(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error (MAE).

    :param np.ndarray actual: Actual data
    :param np.ndarray predicted: Predicted data
    """
    return np.sum(np.abs(actual - predicted)) / len(actual)


def RSE(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Relative Squared Error (RSE).

    :param np.ndarray actual: Actual data
    :param np.ndarray predicted: Predicted data
    """
    rse_n = np.sum((actual - predicted)**2)
    rse_d = np.sum((actual - actual.mean())**2)
    return rse_n / rse_d


def PCC(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Pearson correlation coefficient (PCC)

    :param np.ndarray actual: Actual data
    :param np.ndarray predicted: Predicted data
    """
    pcc_n = len(actual)*np.sum(actual * predicted) - np.sum(actual)*np.sum(predicted)
    pcc_d1 = len(actual)*np.sum(actual**2) - np.sum(actual)**2
    pcc_d2 = len(actual)*np.sum(predicted**2) - np.sum(predicted)**2
    pcc_d = np.sqrt(pcc_d1 * pcc_d2)
    return pcc_n / pcc_d


def R2(actual: np.ndarray, predicted: np.ndarray) -> float:
    """R-squared (R2, R^2).

    :param np.ndarray actual: Actual data
    :param np.ndarray predicted: Predicted data
    """
    r2_n = np.sum((predicted - actual)**2)
    r2_d = np.sum((actual -  actual.mean())**2)
    return 1 - (r2_n / r2_d)


def evaluate_regression(actual: np.ndarray, predicted: np.ndarray):
    """Gets all regression evaluation metrics.

    :param np.ndarray actual: Actual data
    :param np.ndarray predicted: Predicted data
    """
    print(f"Mean Square Error: {round(MSE(actual, predicted), 2)}")
    print(f"Root Mean Square Error: {round(RMSE(actual, predicted), 2)}")
    print(f"Mean Absolute Error: {round(MAE(actual, predicted), 2)}")
    print(f"Relative Square Error: {round(RSE(actual, predicted), 2)}")
    print(f"Pearson Correlation Coeficient: {round(PCC(actual, predicted), 2)}")
    print(f"R-squared: {round(R2(actual, predicted), 2)}")


if __name__ == "__main__":
    # === Data cleaning ===
    data = pd.read_csv("../data/Nasdaq.csv")
    print(data.head())

    # It is necessary to give the correct format to the columns
    data["Date"] = pd.to_datetime(data["Date"])
    data["Close/Last"] = data["Close/Last"].apply(currency_str_to_float)
    data["Open"] = data["Open"].apply(currency_str_to_float)
    data["High"] = data["High"].apply(currency_str_to_float)
    data["Low"] = data["Low"].apply(currency_str_to_float)

    print(data.head())

    # The data is sorted descending, we need them sorted ascending
    data = data.sort_values(by="Date", ascending=True, ignore_index=True)
    print(data.head())
    print(data.describe().T)

    # Does the dataset have missing data?
    print(data.info())
    # R: No, the dataset is complete

    # === Data normalization ===
    # What about the distribution of the data?
    columns = [
        {"name": "Volume", "title": "Volume"},
        {"name": "Open", "title": "Open"},
        {"name": "High", "title": "High"},
        {"name": "Low", "title": "Low"}
    ]

    for column in columns:
        get_distribution_plots(data[column["name"]], column["title"])


    # Columns Close/Last, Open, High and Low present a distribution close to a normal distribution, and their range of 
    # values could be extended in the future. z-score normalization will be used for these columns.
    # Volume present a positive skewed distribution, and its observations show a magnitude of millions.
    # Log normalization will be used for this column.
    data_norm = data.copy()

    scaler = StandardScaler()
    data_norm["Close/Last"] = data_norm["Close/Last"]
    data_norm["Open"] = scaler.fit_transform(data_norm[["Open"]])
    data_norm["High"] = scaler.fit_transform(data_norm[["High"]])
    data_norm["Low"] = scaler.fit_transform(data_norm[["Low"]])
    data_norm["Volume"] = np.log(data_norm["Volume"])

    columns = [
        {"name": "Volume", "title": "Volume_norm"},
        {"name": "Open", "title": "Open_norm"},
        {"name": "High", "title": "High_norm"},
        {"name": "Low", "title": "Low_norm"}
    ]

    for column in columns:
        get_distribution_plots(data_norm[column["name"]], column["title"])

    # Additionally, We want to use month and day as attributes
    data_norm["Day"] = data_norm["Date"].apply(lambda x: x.day)
    data_norm["Month"] = data_norm["Date"].apply(lambda x: x.month)

    print(data_norm.head())
    print(data_norm.describe().T)

    # === Feature selecction ===
    # We want to predict Close/Last using regression, which attributes will be most useful to us?
    get_heatmap(data_norm.drop(columns=["Date"]))
    # R: We will use Open, High, Low and Month as predictors of Close/Last

    date = data_norm[["Date"]].to_numpy()
    Y = data_norm[["Close/Last"]].to_numpy()
    X = data_norm[["Open", "High", "Low", "Month"]].to_numpy()

    fig, axes = plt.subplots(2, 2)
    fig.set_size_inches(14, 8)
    fig.suptitle("Relation between target variable andh choiced features")

    axes[0, 0].scatter(X[:, 0], Y)
    axes[0, 0].set_ylabel("Close/Last")
    axes[0, 0].set_xlabel("Open")

    axes[0, 1].scatter(X[:, 1], Y)
    axes[0, 1].set_ylabel("Close/Last")
    axes[0, 1].set_xlabel("High")

    axes[1, 0].scatter(X[:, 2], Y)
    axes[1, 0].set_ylabel("Close/Last")
    axes[1, 0].set_xlabel("Low")

    axes[1, 1].scatter(X[:, 3], Y)
    axes[1, 1].set_ylabel("Close/Last")
    axes[1, 1].set_xlabel("Month")

    plt.tight_layout()
    plt.savefig(f"./figures/correlations.png", bbox_inches="tight")
    fig.show()

    # Create train and test set
    split_idx = int(np.floor(len(X) * 0.80))
    date_train, date_test = date[:split_idx], date[split_idx:]
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]

    print(f"Total data: {len(Y)}")
    print(f"Train data: {len(Y_train)}")
    print(f"Test data: {len(Y_test)}")

    # === Target variable ===
    # Let's see the evolution of Close/Last over time
    fig = plt.figure(figsize=(8, 4))
    plt.scatter(date_train, Y_train, s=10, label="Train data")
    plt.scatter(date_test, Y_test, s=10, label="Test data")
    plt.title("Close/Last over time")
    plt.xlabel("Date")
    plt.ylabel("Close/Last")
    plt.legend()
    plt.savefig(f"./figures/closelast_time.png", bbox_inches="tight")
    fig.show()

    # === Simple linear regression ===
    choiced_feature = 0

    slr = LinearRegression()
    slr.fit(X_train[:, choiced_feature].reshape(-1, 1), Y_train)
    slr_train = slr.predict(X_train[:, choiced_feature].reshape(-1, 1))
    slr_test = slr.predict(X_test[:, choiced_feature].reshape(-1, 1))

    plot_predictions((date_train, Y_train), (date_test, Y_test), slr_train, slr_test, "Simple linear regression")

    print("===== Simple Linear Regression Evaluation =====")
    evaluate_regression(Y_test, slr_test)

    # === Polynomial regression ===
    choiced_feature = 0
    degree = 2

    pr_feature = PolynomialFeatures(degree=degree, include_bias=False)
    pr_feature = pr_feature.fit_transform(X_train[:, choiced_feature].reshape(-1, 1))

    pr_feature_test = PolynomialFeatures(degree=degree, include_bias=False)
    pr_feature_test = pr_feature_test.fit_transform(X_test[:, choiced_feature].reshape(-1, 1))

    pr = LinearRegression()
    pr.fit(pr_feature, Y_train)
    pr_train = pr.predict(pr_feature)
    pr_test = pr.predict(pr_feature_test)

    plot_predictions((date_train, Y_train), (date_test, Y_test), pr_train, pr_test, "Polynomial regression")

    print("===== Polynomial Regression Evaluation =====")
    evaluate_regression(Y_test, pr_test)

    # === Multiple linear regression ===
    mlr = LinearRegression()
    mlr.fit(X_train, Y_train)
    mlr_train = mlr.predict(X_train)
    mlr_test = mlr.predict(X_test)

    plot_predictions((date_train, Y_train), (date_test, Y_test), mlr_train, mlr_test, "Multiple linear regression")

    print("===== Multiple Linear Regression Evaluation =====")
    evaluate_regression(Y_test, mlr_test)

    # === Ridge regression ===
    rdr = Ridge(alpha=1.0)
    rdr.fit(X_train, Y_train)
    rdr_train = rdr.predict(X_train)
    rdr_test = rdr.predict(X_test)

    plot_predictions((date_train, Y_train), (date_test, Y_test), rdr_train, rdr_test, "Ridge regression")

    print("===== Ridge Regression Evaluation =====")
    evaluate_regression(Y_test, rdr_test)

    input()
