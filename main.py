import timeit
import numpy as np
import matplotlib.pyplot as plt

from data_preprocessor import get_dataframe, get_heatmap, \
    get_drug_moa_pairs, get_train_val_test_data, get_onehot_encode_data
from evaluation import calculate_normalized_mutual_information, compare_clusters_elbow, \
    compare_clusters_normalized_mutual_information, \
    compare_clusters_silhoutte, compare_r_cutoffs
from supervised_learning import run_knn_with_variety_of_k
from pearson_and_spearman_clustering import get_pearson_matrix, \
    get_similarity_labeling_matrix, get_spearman_matrix, group_similar_drugs
from kmeans_clustering import run_kmeans_with_variety_of_k
from two_tier_clustering_with_markov import compare_markov_clustering, get_markov_clusters

filename = 'secondary-screen-dose-response-curve-parameters.csv'

if __name__ == '__main__':
    start = timeit.default_timer()  # timer to keep track of how long each operation takes

    # DATA PREPARATION
    df = get_dataframe(filename)  # get 'auc', 'name', 'ccle_name', 'moa' columns
    print('dataframe ready in ', timeit.default_timer() - start)

    hm = get_heatmap(df, fillNan=True)  # convert df into a heatmap, and fill Nan values if desired
    print('heatmap ready in ', timeit.default_timer() - start)

    drug_list, moa_list, drug_label_map = get_drug_moa_pairs(df)  # get a map from drug to the known moa
    # get training, validation, and test data. The default is: everything is in training data:
    X_train, y_train, X_val, y_val, X_test, y_test = \
        get_train_val_test_data(hm, moa_list, val_ratio=0.15, test_ratio=0.15)
    print('data preparation ready in ', timeit.default_timer() - start)

    # SUPERVISED LEARNING: k-Nearest Neighbors
    knn_accuracy_list = run_knn_with_variety_of_k(X_train, y_train, X_val, y_val, [1, 3, 5, 7, 9])
    print('kNN clustering and evaluation (accuracy) done in ', timeit.default_timer() - start)

    # UNSUPERVISED LEARNING: kMeans
    kmeans_cluster_list = run_kmeans_with_variety_of_k(hm, X_train, X_val, [5, 10, 20, 50, 100, 200, 300, 400, 500])
    kmeans_nmi_list = compare_clusters_normalized_mutual_information(drug_label_map, kmeans_cluster_list)
    print('kMeans clustering and evaluation (NMI) done in ', timeit.default_timer() - start)

    # UNSUPERVISED LEARNING: IMITATING COMPARE ANALYSIS
    corr_m_pearson = get_pearson_matrix(hm)
    corr_m_spearman = get_spearman_matrix(hm)
    print('Pearson and Spearman correlation matrices ready in ', timeit.default_timer() - start)

    r_threshold_list = np.arange(0.3, 1, 0.05)  # list of r-cutoffs to try
    pearson_cluster_list = compare_r_cutoffs(corr_m_pearson, drug_list, r_threshold_list, outcast=True)
    pearson_nmi_list = compare_clusters_normalized_mutual_information(drug_label_map, pearson_cluster_list)
    print('Pearson Clustering and evaluation (NMI) done in ', timeit.default_timer() - start)

    spearman_cluster_list = compare_r_cutoffs(corr_m_spearman, drug_list, r_threshold_list, outcast=True)
    spearman_nmi_list = compare_clusters_normalized_mutual_information(drug_label_map, spearman_cluster_list)
    print('Spearman Clustering and evaluation (NMI) done in ', timeit.default_timer() - start)

    # Commented out silhoutte score due to how slow it is.
    #silhoutte_list = compare_clusters_silhoutte(hm, pearson_cluster_list)
    print('Silhoutte Score Evaluation of Pearson Clustering done in ', timeit.default_timer() - start)

    # Commented out elbow score due to how slow it is.
    #elbow_list = compare_clusters_elbow(hm, pearson_cluster_list)  # 0.5<=r<=0.95, outcast=False
    print('Elbow Technique Evaluation of Pearson Clustering done in ', timeit.default_timer() - start)

    # UNSUPERVISED LEARNING: IMPLEMENTING A UNIQUE TWO-TIER APPROACH WITH MARKOV CLUSTERING
    mcl_nmi_list = compare_markov_clustering(corr_m_pearson, hm, drug_list, drug_label_map, r_threshold_list)
    print('Markov Clustering done in ', timeit.default_timer() - start)
