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
def most_influence_feature_by_pca(datasets, n=25):
    trainings, positives, negatives = datasets
    # t_df = pd.concat(trainings, ignore_index=True)
    initial_feature_names = trainings[0].columns
    dic = dict.fromkeys(range(0, trainings[0].shape[1]), 0)
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


def topic_drop(datasets, topics=None, st=None):
    if topics is None:
        topics = []
    trainings, positives, negatives = datasets
    drp_trainings = list(map(lambda df: df.drop(topics, axis=1), trainings))
    drp_positives = list(map(lambda df: df.drop(topics, axis=1), positives))
    drp_negatives = list(map(lambda df: df.drop(topics, axis=1), negatives))
    if st:
        drp_trainings = list(map(lambda df: df.drop(list(df.filter(regex=st)), axis=1), drp_trainings))
        drp_positives = list(map(lambda df: df.drop(list(df.filter(regex=st)), axis=1), drp_positives))
        drp_negatives = list(map(lambda df: df.drop(list(df.filter(regex=st)), axis=1), drp_negatives))
    return drp_trainings, drp_positives, drp_negatives


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


def find_max_longest_sequence(trainings_names, trainings_preds, alpha=0.95):
    longests = []
    for f, y_pred in zip(trainings_names, trainings_preds):
        longest = M.Measurements.longest_sequence(y_pred, Conf.NEGATIVE_LABEL)
        longests.append(longest)
    longests.sort()
    max_longest = longests[int(round(len(longests) * alpha)) - 1]
    return max_longest


def predictions_information(datasets_preds):
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
    max_longest = find_max_longest_sequence(trainings_names, trainings_preds)
    print("max threshold " + str(max_longest))
    ans += get_acc_info("training sets", zip(trainings_names, trainings_preds), Conf.POSITIVE_LABEL, max_longest) + "\n"
    ans += get_acc_info("positive sets", zip(positives_names, positives_preds), Conf.POSITIVE_LABEL, max_longest) + "\n"
    ans += get_acc_info("negative sets", zip(negatives_names, negatives_preds), Conf.POSITIVE_LABEL, max_longest) + "\n"
    return ans


def dict_sum_preds(multi_paths_preds, max_longest, dict_preds={}):
    for paths_preds in multi_paths_preds:
        for f, y_pred in paths_preds:
            if f not in dict_preds:
                dict_preds[f] = 0
            warning = M.Measurements.count_warnings(y_pred, Conf.NEGATIVE_LABEL, max_longest + 1)
            if warning > 0:
                dict_preds[f] += 1
    return dict_preds


def sum_result(dict):
    tmp = "not exist file"
    count = 0
    anomaly_count = 0
    for key in dict.keys():
        if tmp not in key:
            if not count == 0:
                print(tmp)
                print("number of anomaly: " + str(anomaly_count) + " from " + str(count))
            count = 0
            anomaly_count = 0
            tmp = key[:-6]
        count += 1
        if dict[key] > 0:
            anomaly_count += 1
    print(tmp)
    print("number of anomaly: " + str(anomaly_count) + " from " + str(count))


d = DS.Datasets("data/real_panda/normal/", "data/real_panda/abnormal/")

mic_topics = panda_mic_topics()

mic_dfs = d.get_copied_datasets()
flt_mic_dfs = topics_filter(mic_dfs, mic_topics)
norm_flt_mic_dfs = normalization(flt_mic_dfs)
dfs, n_dfs = run_dbscan_n_predict(norm_flt_mic_dfs)
d.set_predictions(n_dfs[0],n_dfs[1],n_dfs[2])
print(predictions_information(d))
trainings_preds, positives_preds, negatives_preds = d.get_predictions()
trainings_names, positives_names, negatives_names = d.get_names()
multi_paths_preds = [zip(trainings_names, trainings_preds), zip(positives_names, positives_preds),
                     zip(negatives_names, negatives_preds)]
max_longest = find_max_longest_sequence(trainings_names, trainings_preds)
mic_dict_preds = dict_sum_preds(multi_paths_preds, max_longest)
print("summarize Result for Mic features:")
sum_result(mic_dict_preds)

# Macro Anomaly Detection
print("\n\n ------------------------------ PCA ------------------------------------ \n\n")
mac_dfs = d.get_copied_datasets()
d_mac_dfs = topic_drop(mac_dfs, mic_topics, 'Active')
n_d_mac_dfs = normalization(d_mac_dfs)
mac_topics = most_influence_feature_by_pca(n_d_mac_dfs)
flt_n_d_mac_dfs = topics_filter(n_d_mac_dfs, mac_topics)
dfs, n_dfs = run_dbscan_n_predict(flt_n_d_mac_dfs)
d.set_predictions(n_dfs[0], n_dfs[1], n_dfs[2])
print(predictions_information(d))
trainings_preds, positives_preds, negatives_preds = d.get_predictions()
trainings_names, positives_names, negatives_names = d.get_names()
multi_paths_preds = [zip(trainings_names, trainings_preds), zip(positives_names, positives_preds),
                     zip(negatives_names, negatives_preds)]
max_longest = find_max_longest_sequence(trainings_names, trainings_preds)
mac_dict_preds = dict_sum_preds(multi_paths_preds, max_longest, {})
print("summarize Result for Macro features:")
sum_result(mac_dict_preds)





for key in mic_dict_preds.keys():
    mic_dict_preds[key] += mac_dict_preds[key]

print("\n\nUnion Result:")
sum_result(mic_dict_preds)
# # print(flt_dfs)
# no_nan_dfs = removes_nan(dfs)
# n_dfs = normalization(no_nan_dfs)
# most_influence_feature = most_influence_feature_by_pca(n_dfs)
# pca_dfs = pca_filter(n_dfs)
# print(most_influence_feature)
