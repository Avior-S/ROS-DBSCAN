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
from DBSCAN import run_dbscan_n_predict


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
        for i in range(1, 8):
            panda_topics.append('panda_joint' + str(i) + ' ' + epv)
        for i in range(1, 3):
            panda_topics.append('panda_finger_joint' + str(i) + ' ' + epv)
    return panda_topics


def turtlebot3_mic_topics():
    return ['linear velocity x', 'linear velocity y', 'angular velocity z']


def predictions_information(datasets_preds, alpha=0.95):
    def get_acc_info(title, paths_preds, pred_value, max_longest):
        # global an
        ans = "------------------------ %s ------------------------\n" % (title,)
        tmp = ""
        longests = 0
        anomalies = 0
        warnings = 0
        faulty_runs = 0
        acc_size = 5
        f_size = 30
        count = 0
        for f, y_pred in paths_preds:
            count += 1
            y_true = [pred_value] * len(y_pred)
            acc = round(M.Measurements.accuracy_score(y_true, y_pred), acc_size)
            anomalies += len(list(filter(lambda x: x != pred_value, y_pred)))
            longest = M.Measurements.longest_sequence(y_pred, Conf.NEGATIVE_LABEL)
            longests += longest
            warning = M.Measurements.count_warnings(y_pred, Conf.NEGATIVE_LABEL, max_longest + 1)
            if warning > 0:
                faulty_runs += 1
            warnings += warning
            tmp += "%s%s\taccuracy: %s%s\tlongest negative: %s\twarnings = %s\n" % (
            f, " " * (f_size - (len(f) - 2)), acc, " " * (acc_size - (len(str(acc)) - 2)), longest, warning)
            # M.Measurements.draw_chart(f, options.charts + f[:-4],y_pred,Conf.NEGATIVE_LABEL)
        # an = longests/len(paths_preds)
        ans = ans + "longest average = %s, anomalies = %s,  total warnings = %s,  faulty runs = %s,  non-faulty runs = %s\n" % (
        longests / count, anomalies, warnings, faulty_runs, count - faulty_runs)
        return ans + tmp

    trainings_preds, positives_preds, negatives_preds = datasets_preds.get_predictions()
    trainings_names, positives_names, negatives_names = datasets_preds.get_names()
    ans = ""
    longests = []
    for f, y_pred in zip(trainings_names, trainings_preds):
        longest = M.Measurements.longest_sequence(y_pred, Conf.NEGATIVE_LABEL)
        longests.append(longest)
    longests.sort()
    max_longest = longests[int(round(len(longests) * alpha)) - 1]
    print("max threshold " + str(max_longest))
    ans += get_acc_info("training sets", zip(trainings_names, trainings_preds), Conf.POSITIVE_LABEL, max_longest) + "\n"
    ans += get_acc_info("positive sets", zip(positives_names, positives_preds), Conf.POSITIVE_LABEL, max_longest) + "\n"
    ans += get_acc_info("negative sets", zip(negatives_names, negatives_preds), Conf.POSITIVE_LABEL, max_longest) + "\n"
    return ans


d = DS.Datasets("data/real_panda/normal/", "data/real_panda/test/")
dfs = d.get_datasets()
topics = panda_mic_topics()
flt_dfs = topics_filter(dfs, topics)
dfs, n_dfs = run_dbscan_n_predict(flt_dfs)
d.update_predictions(n_dfs[0],n_dfs[1],n_dfs[2])
print(predictions_information(d))
# print(flt_dfs)
# no_nan_dfs = removes_nan(dfs)
# n_dfs = normalization(no_nan_dfs)
# most_influence_feature = most_influence_feature_by_pca(n_dfs)
# pca_dfs = pca_filter(n_dfs)
# print(most_influence_feature)
