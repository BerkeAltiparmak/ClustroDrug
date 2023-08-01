import math

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from pearson_and_spearman_clustering import group_similar_drugs, get_similarity_labeling_matrix


def compare_r_cutoffs(corr_m, drug_list, r_cutoff_list, want_supercluster=False, outcast=False):
    rcutoff_clustered_drugs_list = []
    for r_cutoff in r_cutoff_list:
        initial_cluster, _ = get_similarity_labeling_matrix(corr_m, drug_list, r_cutoff)
        clustered_drugs = group_similar_drugs(initial_cluster, 2, has_outcast_cluster=outcast)
        #if want_supercluster:
            #clustered_drugs = get_super_clustered_drugs(clustered_drugs)
        rcutoff_clustered_drugs_list.append(clustered_drugs)

    return rcutoff_clustered_drugs_list


def compare_clusters_silhoutte(hm, rcutoff_clustered_drugs_list):
    silhoutte_list = []
    for clustered_drugs in rcutoff_clustered_drugs_list:
        if len(clustered_drugs) != 0:
            cluster_heatmap, cluster_label_list = get_repeated_dataset(clustered_drugs, hm)
            silhoutte_list.append(get_silhoutte_scores(cluster_heatmap, cluster_label_list))
        else:
            silhoutte_list.append(0)

    return silhoutte_list


def get_silhoutte_scores(cluster_heatmap, cluster_label_list):
    score = silhouette_score(cluster_heatmap, cluster_label_list, metric='euclidean')
    return score


def compare_clusters_elbow(hm, rcutoff_clustered_drugs_list):
    elbow_list = []
    for clustered_drugs in rcutoff_clustered_drugs_list:
        if len(clustered_drugs) != 0:
            elbow_list.append(get_elbow_score(hm, clustered_drugs))
        else:
            elbow_list.append(0)

    return elbow_list


def get_elbow_score(hm, clustered_drugs):
    curr_wss = 0
    for cluster in clustered_drugs:
        cluster_heatmap, _ = get_repeated_dataset([cluster], hm)
        mean_cell_profile = np.array(cluster_heatmap.mean(axis=0))
        for drug in cluster:
            drug_cell_profile = np.array(cluster_heatmap.loc[drug])
            curr_wss += np.linalg.norm(mean_cell_profile - drug_cell_profile) ** 2
    return curr_wss


def get_repeated_dataset(drug_list_in_cluster, hm):
    # df = pd.DataFrame(columns=hm.columns)
    auc_list = []
    name_list = []
    cluster_label_list = []
    cluster_label = 0
    for cluster in drug_list_in_cluster:
        for drug in cluster:
            auc_list.append(hm.loc[drug])
            name_list.append(hm.loc[drug].name)
            cluster_label_list.append(cluster_label)
        cluster_label += 1
    df = pd.DataFrame(auc_list, columns=hm.columns, index=name_list)
    return (df, cluster_label_list)


def get_label_count_map(drug_list, drug_label_map):
    label_count_map = {}
    for drug_name in drug_list:
        label = drug_label_map[drug_name]
        if label in label_count_map:
            label_count_map[label] += 1
        else:
            label_count_map[label] = 1

    return label_count_map


def get_class_labels_entropy(total_nonunique_labels, drugs_label_count_map):
    # total_nonunique_labels = len(drug_classes_map)
    H_y = 0
    for label in drugs_label_count_map:
        label_prop = drugs_label_count_map[label] / total_nonunique_labels
        H_y += -label_prop * math.log2(label_prop)
    return H_y


def get_cluster_labels_entropy(total_nonunique_labels, clustered_drugs):
    H_c = 0
    for cluster in clustered_drugs:
        cluster_prop = len(cluster) / total_nonunique_labels
        H_c += -cluster_prop * math.log2(cluster_prop)
    return H_c


def get_mutual_information_between_classes_and_clusters(class_labels_entropy, drug_classes_map, total_nonunique_labels,
                                                        clustered_drugs):
    H_y_c = 0
    H_y = class_labels_entropy
    for cluster in clustered_drugs:
        cluster_prop = len(cluster) / total_nonunique_labels
        label_count_within_cluster_map = {}
        for drug in cluster:
            label = drug_classes_map[drug]
            if label in label_count_within_cluster_map:
                label_count_within_cluster_map[label] += 1
            else:
                label_count_within_cluster_map[label] = 1
        H_y_within_cluster = get_class_labels_entropy(len(cluster), label_count_within_cluster_map)
        H_y_c += cluster_prop * H_y_within_cluster

    I_y_c = H_y - H_y_c

    return I_y_c


def calculate_normalized_mutual_information(drug_label_map, clustered_drugs):
    all_drugs_in_cluster = [item for sublist in clustered_drugs for item in sublist]
    label_count_map = get_label_count_map(all_drugs_in_cluster, drug_label_map)
    total_nonunique_labels = len(all_drugs_in_cluster)
    H_y = get_class_labels_entropy(total_nonunique_labels, label_count_map)
    H_c = get_cluster_labels_entropy(total_nonunique_labels, clustered_drugs)
    I_y_c = get_mutual_information_between_classes_and_clusters(H_y, drug_label_map, total_nonunique_labels,
                                                                clustered_drugs)

    NMI_y_c = 0
    if H_y * H_c > 0:
        NMI_y_c = 2 * I_y_c / (H_y + H_c)

    return NMI_y_c


def compare_clusters_normalized_mutual_information(drug_label_map, rcutoff_clustered_drugs_list):
    nmi_list = []
    for clustered_drugs in rcutoff_clustered_drugs_list:
        nmi = calculate_normalized_mutual_information(drug_label_map, clustered_drugs)
        nmi_list.append(nmi)

    return nmi_list


def get_levenshtein_distance_of_lists(list1, list2):
    list1_has = []
    intersection = []
    list2_has = []
    distance = 0
    for e1 in list1:
        if e1 not in list2:
            distance += 1
            list1_has.append(e1)
        else:
            intersection.append(e1)
    for e2 in list2:
        if e2 not in list1:
            distance += 1
            list2_has.append(e2)

    return (distance, list1_has, intersection, list2_has)
