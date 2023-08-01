"""
Preprocess the drug/cell-line file into DataFrames, Heatmaps, and
Training, Validation, and Test datasets.
"""
from typing import Any, Dict, Tuple

import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)  # Otherwise Pandas throws warning


def get_dataframe(filename: str) -> DataFrame:
    """
    Get Pandas dataframe of a file with columns 'auc', 'name', 'ccle_name', 'moa' columns,
    such as the 'secondary-screen-dose-response-curve-parameters.csv' file

    :param filename: Name of the file to create a dataframe from
    :return: Pandas dataframe with columns 'auc', 'name', 'ccle_name', 'moa'
    """
    df = pd.read_csv(filename)
    df_essentials = df[['auc', 'name', 'ccle_name', 'moa']]

    return df_essentials


def get_heatmap(df: DataFrame, fillNan=False) -> DataFrame:
    """
    Convert the DataFrame into a Heatmap of auc values, with drugs as rows and cell lines as columns

    :param df: Pandas dataframe with columns at least 'auc', 'name', 'ccle_name'
    :param fillNan: Whether or not to fill Nan values
    :return: Heatmap of auc values, with drugs as rows and cell lines as columns
    """
    hm = pd.pivot_table(df, values='auc', index=['name'], columns='ccle_name')
    if fillNan:
        hm = fill_nan_values(hm)
    return hm


def fill_nan_values(hm: DataFrame):
    """
    Fill Nan value of an entry by taking the average of the column (average of drugs' auc values on
    the cell line that corresponds to the Nan entry) and the average of the row (average of the auc of the
    drug that corresponds to the Nan entry), and then taking the average of these two.
    Example: X = [[Nan, 1, 2],
                    [3, 4, 5],
                    [6, 7, 8]]
    fill_nan_values(X) = [[3, 1, 2],
                          [3, 4, 5],
                          [6, 7, 8]]

    :param hm: Heatmap of auc values
    :return: Heatmap of auc values, with Nan values imputated.
    """
    rhm = hm.apply(lambda row: row.fillna(row.mean()), axis=1)
    chm = hm.fillna(hm.mean())
    return (chm + rhm) / 2


def get_drug_moa_pairs(df: DataFrame) -> tuple[Any, Any, dict[Any, Any]]:
    """
    Returns the map from drug to the moa that corresponds to that drug according to the df.

    :param df: Pandas dataframe with columns at least 'name', 'moa'
    :return: list of drugs, list of moas, map from drug to moa
    """
    df_needed = df[['name', 'moa']].drop_duplicates().sort_values('name')
    drug_list = df_needed['name'].tolist()
    moa_list = df_needed['moa'].tolist()
    drug_moa_dict = dict()
    for i in range(len(drug_list)):
        drug_moa_dict[drug_list[i]] = moa_list[i]

    return (drug_list, moa_list, drug_moa_dict)


def get_train_val_test_data(hm: DataFrame, moa_list: list, val_ratio=0.15, test_ratio=0.15):
    """
    Splits the data into train, validation, and test datasets.

    :param hm: Heatmap (DataFrame) of drugs and cell lines, with auc values
    :param moa_list: List of moas of drugs, where order is important
    :param val_ratio: Ratio of the length of validation set to the whole data
    :param test_ratio: Ratio of the length of test set to the whole data
    :return: Train, validation, and test datasets, each with both X and y data.
    """
    X = hm.values
    y = moa_list

    if test_ratio == 0:
        X_train, X_test, y_train, y_test = X, np.array([]), y, []
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=6)

    if val_ratio == 0:
        X_train, X_val, y_train, y_val = X_train, np.array([]), y_train, []
    else:
        X_train, X_val, y_train, y_val = \
            train_test_split(X_train, y_train, test_size=(val_ratio / (1 - test_ratio)), random_state=6)

    return (X_train, y_train, X_val, y_val, X_test, y_test)


def get_onehot_encode_data(data: list):
    """
    Onehot encodes the data, which might be useful for y (moa_list) if Neural Network
    is used as a model. This function is not used because we got rid off the NN approach.

    :param data: List of strings
    :return: Onehot encoded version of the data
    """
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(data)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    return onehot_encoded


def get_pancreas_heatmap(filename='clean_wide.csv') -> DataFrame:
    """
    Get Pandas dataframe of a specific Notta Lab pancreas data.

    :param filename: Name of the Notta Lab file
    :return: Pandas Dataframe with drugs as the row, cell lines as the columns.
    """
    df = pd.read_csv(filename)
    df = df.drop('Location', axis=1)
    df = df.drop('hits', axis=1)
    df.index = df['drug']
    df = df.drop('drug', axis=1)

    return df
