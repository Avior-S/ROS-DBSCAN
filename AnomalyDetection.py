import Dataset as DS
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from Style import Configure as Conf
import Measurements as M
import numpy as np
from scipy.spatial.distance import cdist


def feature_selection(feature_func, datasets):
    new_datasets = []
    for dataset in datasets:
        new_datasets.append(feature_func(dataset))
    return new_datasets


def removes_nan(datasets):
    new_datasets = []
    for dataset in datasets:
        new_datasets.append(list(map(lambda df: df.fillna(0), dataset)))
    return new_datasets


def normalization(datasets):
    trainings, positives, negatives = datasets
    t_df = pd.concat(trainings, ignore_index=True)
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(t_df)
#     n_trainings = list(map(lambda df: ((df-df.min())/(df.max()-df.min())).fillna(0), trainings))
#     n_positives = list(map(lambda df: ((df-df.min())/(df.max()-df.min())).fillna(0), positives))
#     n_negatives = list(map(lambda df: ((df-df.min())/(df.max()-df.min())).fillna(0), negatives))
    n_trainings = list(map(lambda df: pd.DataFrame(min_max_scaler.transform(df)), trainings))
    n_positives = list(map(lambda df: pd.DataFrame(min_max_scaler.transform(df)), positives))
    n_negatives = list(map(lambda df: pd.DataFrame(min_max_scaler.transform(df)), negatives))
    return n_trainings, n_positives, n_negatives


def pca_filter(datasets, n=30):
    trainings, positives, negatives = datasets
    t_df = pd.concat(trainings, ignore_index=True)
    pca = PCA(n_components=n)
    pca = pca.fit(t_df)
    ipca_training = list(map(lambda df: pca.transform(df), trainings))
    ipca_positive = list(map(lambda df: pca.transform(df), positives))
    ipca_negative = list(map(lambda df: pca.transform(df), negatives))
    return ipca_training, ipca_positive, ipca_negative


d = DS.Datasets("data/real_panda/normal/", "data/real_panda/test/")
dfs = d.get_datasets()
no_nan_dfs = removes_nan(dfs)
n_dfs = normalization(no_nan_dfs)
pca_dfs = pca_filter(n_dfs)
print(pca_dfs)
