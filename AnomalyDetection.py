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
from operator import itemgetter


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

    n_trainings = list(map(lambda df: pd.DataFrame(min_max_scaler.transform(df), columns=t_df.columns), trainings))
    n_positives = list(map(lambda df: pd.DataFrame(min_max_scaler.transform(df), columns=t_df.columns), positives))
    n_negatives = list(map(lambda df: pd.DataFrame(min_max_scaler.transform(df), columns=t_df.columns), negatives))
    return n_trainings, n_positives, n_negatives


# The important features are the ones that influence more the components and thus,
# have a large absolute value/score on the component.
def most_influence_feature_by_pca(datasets, n=30):
    trainings, positives, negatives = datasets
    t_df = pd.concat(trainings, ignore_index=True)
    initial_feature_names = t_df.columns
    dic = dict.fromkeys(range(0, t_df.shape[1]), 0)
    for training in trainings:
        pca = PCA(n_components=n)
        pca = pca.fit(training)
        for arr in pca.components_:
            abs_arr = np.abs(arr)
            for i in range(len(abs_arr)):
                dic[i] += abs_arr[i]
    dic_most_important_feature = dict(sorted(dic.items(), key=itemgetter(1), reverse=True)[:n])
    most_important_index = list(dic_most_important_feature.keys())
    # get the names
    most_important_names = [initial_feature_names[most_important_index[i]] for i in range(n)]
    # ipca_training = list(map(lambda df: pca.transform(df), trainings))
    # ipca_positive = list(map(lambda df: pca.transform(df), positives))
    # ipca_negative = list(map(lambda df: pca.transform(df), negatives))
    return most_important_names


def topic_drop(datasets, topics=None):
    if topics is None:
        topics = []
    trainings, positives, negatives = datasets
    drp_training = list(map(lambda df: df.drop(topics, axis=1), trainings))
    drp_positive = list(map(lambda df: df.drop(topics, axis=1), positives))
    drp_negative = list(map(lambda df: df.drop(topics, axis=1), negatives))
    return drp_training, drp_positive, drp_negative


def topics_filter(datasets, topics=None):
    if topics is None:
        topics = []
    trainings, positives, negatives = datasets
    flt_training = list(map(lambda df: df[topics], trainings))
    flt_positive = list(map(lambda df: df[topics], positives))
    flt_negative = list(map(lambda df: df[topics], negatives))
    return flt_training, flt_positive, flt_negative


def panda_mic_topics():
    panda_topics = []
    for epv in ['effort', 'position', 'velocity']:
        for i in range(1, 7):
            panda_topics.append('panda_joint' + str(i) + ' ' + epv)
        for i in range(1, 2):
            panda_topics.append('panda_finger_joint' + str(i) + ' ' + epv)
    return panda_topics


def turtlebot3_mic_topics():
    return ['linear velocity x', 'linear velocity y', 'angular velocity z']


d = DS.Datasets("data/real_panda/normal/", "data/real_panda/test/")
dfs = d.get_datasets()
# topics = panda_mic_topics()
# flt_dfs = topics_filter(dfs, topics)
# print(flt_dfs)
no_nan_dfs = removes_nan(dfs)
n_dfs = normalization(no_nan_dfs)
most_influence_feature = most_influence_feature_by_pca(n_dfs)
# pca_dfs = pca_filter(n_dfs)
print(most_influence_feature)
