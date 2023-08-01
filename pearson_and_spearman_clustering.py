import numpy as np
from scipy.stats import stats


def get_pearson_matrix(hm):
    """
    Get the Pearson Correlation Coefficient matrix of the drug profiles in the given heatmap.

    :param hm: Heatmap (DataFrame) of drugs, to find the name of the drugs.
    :return pearson_matrix: Pearson Correlation Coefficient matrix of drugs.
    """
    pearson_matrix = np.corrcoef(hm)
    return pearson_matrix


def get_spearman_matrix(hm):
    """
    Get the Spearman's Rank Correlation Coefficient matrix of the drug profiles in the given heatmap.

    :param hm: Heatmap (DataFrame) of drugs, to find the name of the drugs
    :return spearman_matrix: Spearman's Rank Correlation Coefficient matrix of drugs
    """
    spearman_matrix, _ = stats.spearmanr(hm, axis=1)
    return spearman_matrix


def compare_COMPARE_Analysis(pearson_matrix, drug_list, r_threshold_list):
    """
    Run Pearson Algorithm for a range of r_cutoff values.

    :param pearson_matrix: Pearson Correlation Coefficient Matrix.
    :param drug_list: List of drugs in the data.
    :param r_threshold_list: List of r_cutoff values wanted.
    :return clustered_drugs_list: List of clusters of drugs computed from each r_cutoff value.
    """
    clustered_drugs_list = []
    for r_threshold in r_threshold_list:
        initial_cluster, _ = get_similarity_labeling_matrix(pearson_matrix, drug_list, r_threshold)
        clustered_drugs = group_similar_drugs(initial_cluster, minimum_cluster_length=2, has_outcast_cluster=False)
        clustered_drugs_list.append(clustered_drugs)

    return clustered_drugs_list


def get_similarity_labeling_matrix(corr_m, drug_list, r_cutoff=0.6):
    """
    Put two drugs in the same list if they have a similarity above the threshold provided.

    :param corr_m: Correlation Matrix.
    :param drug_list: List of drugs.
    :param r_cutoff: The threshold to consider when concluding whether two drugs are similar enough.
    :return: List of initial clusters and Correlation Matrix with values less than r_cutoff rounded
    to zero.
    """
    initial_cluster = []
    modified_similarity_matrix = []
    for row_i in corr_m:
        row_i_similardrugs_set = []
        row_i_similarscore_set = []
        for drug_j_index in range(0, len(row_i)):
            drug_j_similarity_with_drug_i = row_i[drug_j_index]
            if drug_j_similarity_with_drug_i >= r_cutoff:
                row_i_similardrugs_set.append(drug_list[drug_j_index])
                row_i_similarscore_set.append(drug_j_similarity_with_drug_i)
            else:
                row_i_similarscore_set.append(0)
        initial_cluster.append(row_i_similardrugs_set)
        modified_similarity_matrix.append(row_i_similarscore_set)

    return (initial_cluster, np.array(modified_similarity_matrix))


def group_similar_drugs(initial_cluster, minimum_cluster_length=2, has_outcast_cluster=False):
    """
    Remove same-but-different-ordered clusters and also clusters with length less than the minimum
    required. If outcast argument is True, then instead of removing the individual clusters, merge
    them into one big outcast cluster. Hence, outcast cluster consists of drugs that are not similar
    enough to any other drug in the data. Return the list of clusters.

    :param initial_cluster: Rough clusters computed from get_similarity_labeling_matrix.
    :param minimum_cluster_length: Minimum number of drugs to have in a cluster.
    :param has_outcast_cluster: If true, merge clusters that do not meet the minimum_cluster_length
    criteria, useful when computing NMI.
    :return: List of actual clusters.
    """
    clustered_drugs = []
    outcast_cluster = []
    for label in initial_cluster:
        sorted_label = sorted(label)  # to check uniqueness in the list
        if len(label) >= minimum_cluster_length and sorted_label not in clustered_drugs:
            clustered_drugs.append(sorted_label)
        elif 0 < len(label) < minimum_cluster_length and has_outcast_cluster:
            outcast_cluster.extend(label)

    if has_outcast_cluster and len(outcast_cluster) > 0:
        clustered_drugs.append(outcast_cluster)

    return clustered_drugs


def get_super_clustered_drugs(clustered_drugs):
    """
    Compute and return superclusters. Supercluster are the merge of clusters who share at least
    one drug in common

    :param clustered_drugs: The cluster of drugs that are already computed.
    :return: Superclusters of drugs.
    """
    super_clustered_drugs = []
    for cluster1 in clustered_drugs:
        super_cluster = cluster1
        i = 0
        while i < len(clustered_drugs):
            cluster2 = clustered_drugs[i]
            i += 1
            union_set = set(super_cluster + cluster2)
            union_list = list(union_set)

            # if they have shared elements but one set is not the other set's subset
            if len(union_list) < len(super_cluster) + len(cluster2) and \
                    not (set(cluster2).issubset(set(super_cluster))):
                super_cluster = union_list
                i = 0
        super_clustered_drugs.append(super_cluster)

    return group_similar_drugs(super_clustered_drugs, minimum_cluster_length=3, has_outcast_cluster=False)
