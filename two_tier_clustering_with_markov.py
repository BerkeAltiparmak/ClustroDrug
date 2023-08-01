import markov_clustering as mcl

from evaluation import calculate_normalized_mutual_information
from pearson_and_spearman_clustering import get_similarity_labeling_matrix, group_similar_drugs


def get_markov_clusters(similarity_matrix, hm, inflation=2.0, expansion=2, pruning_threshold=0.001):
    """
    Runs Markov Clustering Algorithm on the given similarity matrix, and returns the resulting adjacency matrix,
    the clusters identified by Markov Clustering,

    :param similarity_matrix: Similarity Matrix
    :param hm: Heatmap of drugs
    :param inflation:
    :param expansion:
    :param pruning_threshold:
    :return:
    """
    mcl_result = mcl.run_mcl(similarity_matrix, inflation=inflation, expansion=expansion,
                             pruning_threshold=pruning_threshold)
    clusters = mcl.get_clusters(mcl_result)

    mcl_cluster_index_list = []
    mcl_cluster_list = []
    for c in clusters:
        index_list = list(c)
        mcl_cluster_index_list.append(index_list)
        mcl_cluster_list.append(list(hm.iloc[index_list].index))

    return (mcl_result, mcl_cluster_list, mcl_cluster_index_list)


def compare_markov_clustering(corr_m_pearson, hm, drug_list, drug_label_map, r_threshold_list):
    mcl_nmi_list = []
    for r in r_threshold_list:
        _, pearson_matrix = get_similarity_labeling_matrix(corr_m_pearson, drug_list, r)
        mcl_matrix, mcl_cluster, _ = get_markov_clusters(pearson_matrix, hm, 2, 2)
        mcl_final_cluster = group_similar_drugs(mcl_cluster, 2, has_outcast_cluster=False)
        mcl_nmi = calculate_normalized_mutual_information(drug_label_map, mcl_final_cluster)
        mcl_nmi_list.append(mcl_nmi)

    return mcl_nmi_list
