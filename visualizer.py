import json

import networkx as nx
from pyvis.network import Network
import numpy as np
import pandas as pd
import seaborn as sns

from evaluation import get_levenshtein_distance_of_lists
from pearson_and_spearman_clustering import get_super_clustered_drugs


def plot_heatmaps_of_wanted_clusters(hm, drug_label_map, clustered_drugs, path="heatmaps/",
                                     want_super_clustered_drugs=True,
                                     wanted_cluster_length=3, has_outcast_cluster_for_all=False):
    if has_outcast_cluster_for_all:
        clustered_drugs = clustered_drugs.copy()[:-1]
    label_occurrance_map = {}
    count = 0
    for cluster in clustered_drugs:
        if len(cluster) >= wanted_cluster_length:
            hm_cluster, _ = get_repeated_dataset([cluster], hm)

            sns.set(font_scale=0.2)
            cm = sns.clustermap(hm_cluster, metric="euclidean", method="weighted", cmap="RdBu",
                                robust='TRUE', dendrogram_ratio=(0.05, 0.1), vmin=0, vmax=1,
                                xticklabels=1, yticklabels=1, figsize=(100, 5),
                                cbar_pos=(0, 0.1, 0.02, 0.3))
            fig = cm._figure

            filename = ""
            if drug_label_map != []:
                label_count_map = get_label_count_map(cluster, drug_label_map)
                most_occurring_label = max(label_count_map, key=label_count_map.get)
                most_occurring_label_count = label_count_map[most_occurring_label]
                label_info = str(most_occurring_label) + "-count-" + \
                             str(most_occurring_label_count) + "_"
                if most_occurring_label not in label_occurrance_map:
                    label_occurrance_map[most_occurring_label] = 1
                    filename = label_info + "cluster_rcutoff_0.6" + ".pdf"
                else:
                    label_occurrance_map[most_occurring_label] += 1
                    filename = label_info + str(label_occurrance_map[most_occurring_label]) \
                               + "_cluster_rcutoff_0.6" + ".pdf"
            else:
                filename = path[:-9] + "cluster_" + str(count) + ".pdf"
                count += 1

            fig.savefig(filename)

    if want_super_clustered_drugs:
        plot_heatmaps_of_wanted_clusters(hm, drug_label_map,
                                         get_super_clustered_drugs(clustered_drugs),
                                         path=path + "AAsuperclusters/",
                                         want_super_clustered_drugs=False,
                                         has_outcast_cluster_for_all=False)


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


def get_drug_occurance_count(clustered_drugs):
    drug_occurance_count = {}
    for c in clustered_drugs:
        for d in c:
            if d not in drug_occurance_count:
                drug_occurance_count[d] = 1
            else:
                drug_occurance_count[d] += 1

    return drug_occurance_count


def get_label_count_map(drug_list, drug_label_map):
    label_count_map = {}
    for drug_name in drug_list:
        label = drug_label_map[drug_name]
        if label in label_count_map:
            label_count_map[label] += 1
        else:
            label_count_map[label] = 1

    return label_count_map


def get_matrix_from_clusters(drug_list, clusters: list[list[str]]):
    """
    Computes adjacency matrix from the list of clusters of drugs.

    :param drug_list: List of drugs in the data, order matters.
    :param clusters: List of clusters of drugs identified by one of the models.
    :return: The adjacency matrix of the model's output.
    """
    cluster_matrix = np.zeros((len(drug_list), len(drug_list)))

    for c in clusters:
        for d1 in c:
            for d2 in c:
                d1_index = drug_list.index(d1)
                d2_index = drug_list.index(d2)
                cluster_matrix[d1_index][d2_index] = 1
                cluster_matrix[d2_index][d1_index] = 1

    return cluster_matrix


def convert_map_to_json(python_map, filename="pearson_vs_spearman_0.6-r-cutoff.json"):
    with open(filename, "w") as outfile:
        json.dump(python_map, outfile, indent=4, sort_keys=False)


def get_A_vs_B(hm, drug_label_map, clustered_drugs_A, clustered_drugs_B):
    superclusters_A = get_super_clustered_drugs(clustered_drugs_A)
    superclusters_B = get_super_clustered_drugs(clustered_drugs_B)
    filename_list_A = temp(hm, drug_label_map, superclusters_A)
    filename_list_B = temp(hm, drug_label_map, superclusters_B)
    return compare_superclusters(superclusters_A, superclusters_B,
                                 filename_list_A, filename_list_B)


def temp(hm, drug_label_map, clustered_drugs, path="heatmaps/",
         want_super_clustered_drugs=True,
         wanted_cluster_length=3, has_outcast_cluster_for_all=False):
    if has_outcast_cluster_for_all:
        clustered_drugs = clustered_drugs.copy()[:-1]
    label_occurrance_map = {}
    filename_list = []
    for cluster in clustered_drugs:
        if len(cluster) >= wanted_cluster_length:
            hm_cluster, _ = get_repeated_dataset([cluster], hm)

            label_count_map = get_label_count_map(cluster, drug_label_map)
            most_occurring_label = max(label_count_map, key=label_count_map.get)
            most_occurring_label_count = label_count_map[most_occurring_label]
            label_info = str(most_occurring_label) + "-count-" + \
                         str(most_occurring_label_count) + "_"
            filename = ""
            if most_occurring_label not in label_occurrance_map:
                label_occurrance_map[most_occurring_label] = 1
                filename = label_info + "cluster_rcutoff_0.6" + ".pdf"
            else:
                label_occurrance_map[most_occurring_label] += 1
                filename = label_info + str(label_occurrance_map[most_occurring_label]) \
                           + "_cluster_rcutoff_0.6" + ".pdf"
            filename_list.append(filename)
    return filename_list


def compare_superclusters(superclusters1, superclusters2, pearson_filename_list, spearman_filename_list):
    comparison_array = []
    difference_map = {}
    for scluster1 in superclusters1:
        comparison_for_scluster1 = []
        for scluster2 in superclusters2:
            comparison_score, pearson_has, intersection, spearman_has = \
                get_levenshtein_distance_of_lists(scluster1, scluster2)
            comparison_score = comparison_score / (len(scluster1) + len(scluster2))
            comparison_for_scluster1.append(comparison_score)
            if comparison_score < 1:
                map_name = pearson_filename_list[superclusters1.index(scluster1)] + "__VS__" + \
                           spearman_filename_list[superclusters2.index(scluster2)]
                difference_map[map_name] = \
                    {"Pearson exclusive": pearson_has,
                     "Spearman exclusive": spearman_has,
                     "Intersection": intersection}

        comparison_array.append(comparison_for_scluster1)

    comparison_df = pd.DataFrame(comparison_array, columns=spearman_filename_list, index=pearson_filename_list)

    return comparison_df, difference_map


def get_comparison_graphs(drug_list, pearson_matrix, spearman_matrix, mcl_result, isDynamic=True):
    """
    Compares three models' clusters by displaying each model's clusters at the same time by color
    coding it so that we know which cluster is identified by which model.

    :param drug_list: List of drugs in the data, order matters.
    :param pearson_matrix: Adjacency matrix of the pearson algorithm.
    :param spearman_matrix: Adjacency matrix of the spearman algorithm.
    :param mcl_result: Adjacency matrix of the MCL algorithm.
    :param isDynamic: Whether or not the visual should be dynamic.
    :return:
    """
    for i in range(0, len(pearson_matrix)):
        pearson_matrix[i][i] = 0  # making the diagonal zero instead of 1s to remove self loops.
    for i in range(0, len(mcl_result)):
        mcl_result[i][i] = 0  # making the diagonal zero instead of 1s to remove self loops.
    for i in range(0, len(spearman_matrix)):
        spearman_matrix[i][i] = 0  # making the diagonal zero instead of 1s to remove self loops.

    graph_matrix = (pearson_matrix + spearman_matrix) / \
                   (np.ceil(pearson_matrix) + np.ceil(spearman_matrix) +
                    ((np.ceil(pearson_matrix) + np.ceil(spearman_matrix)) == 0) * 1)
    df_graph = pd.DataFrame(graph_matrix, index=drug_list, columns=drug_list)
    G_mcl = nx.from_pandas_adjacency(pd.DataFrame(mcl_result, index=drug_list, columns=drug_list))
    G_pearson = nx.from_pandas_adjacency(pd.DataFrame(pearson_matrix, index=drug_list, columns=drug_list))
    G_spearman = nx.from_pandas_adjacency(pd.DataFrame(spearman_matrix, index=drug_list, columns=drug_list))
    G_comp = nx.from_pandas_adjacency(df_graph)
    G_comp.remove_nodes_from(list(nx.isolates(G_comp)))
    edges = list(G_comp.edges())
    mcl_edges = list(G_mcl.edges())
    pearson_edges = list(G_pearson.edges())
    spearman_edges = list(G_spearman.edges())
    weights = [G_comp[u][v]['weight'] for u, v in edges]
    color_scheme = [
        [['black', 'red'], ['blue', 'purple']],  # 000 is impossible, 001 is red, 010 is blue, 011 is purple
        [['green', 'yellow'], ['cyan', 'white']]  # 100 is green, 101 is yellow, 110 is cyan, 111 is black
    ]
    colors = [color_scheme[(u, v) in spearman_edges]
              [(u, v) in mcl_edges]
              [(u, v) in pearson_edges] for u, v in edges]  # 1xx is spearman, x1x is mcl, xx1 is person
    norm_weight = [(float(i) - min(weights) + 0.1) / (max(weights) - min(weights) + 0.1) for i in weights]
    G_comp2 = nx.from_pandas_adjacency(pd.DataFrame())
    for i in range(0, len(edges)):
        u, v = edges[i]
        G_comp2.add_edge(u, v, color=colors[i], weight=norm_weight[i] * 5, title=weights[i])

    nt = Network('700px', '1400px', bgcolor="#222222", font_color="white", select_menu=True)
    nt.from_nx(G_comp2, show_edge_weights=True)

    return (nt, G_comp2)
