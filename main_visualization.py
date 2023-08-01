import timeit
from data_preprocessor import get_drug_moa_pairs, get_heatmap, get_dataframe
from pearson_and_spearman_clustering import get_pearson_matrix, get_spearman_matrix, group_similar_drugs, \
    get_similarity_labeling_matrix
from two_tier_clustering_with_markov import get_markov_clusters
from visualizer import convert_map_to_json, get_A_vs_B, get_comparison_graphs, plot_heatmaps_of_wanted_clusters

filename = 'secondary-screen-dose-response-curve-parameters.csv'

if __name__ == '__main__':
    start = timeit.default_timer()  # timer to keep track of how long each operation takes

    df = get_dataframe(filename)  # get 'auc', 'name', 'ccle_name', 'moa' columns
    print('dataframe ready in ', timeit.default_timer() - start)

    hm = get_heatmap(df, fillNan=True)  # convert df into a heatmap, and fill Nan values if desired
    print('heatmap ready in ', timeit.default_timer() - start)

    drug_list, moa_list, drug_label_map = get_drug_moa_pairs(df)  # get a map from drug to the known moa
    print('data preparation ready in ', timeit.default_timer() - start)

    # Visualize the operation you want.
    while True:
        operation = input("Type in the visualization operation you want to do."
                          " Options: heatmaps, comparison_json, graph")

        if operation == "heatmaps":
            folder_name = input("Give a name for the folder to store heatmaps. Ex: heatmaps")
            folder_name += '/'

            algorithm = input("Type p for Pearson, s for Spearman, "
                              "pm for Pearson+Markov Clustering, sm for Spearman+Markov Clustering.")
            r_cutoff = float(input("Choose an r_cutoff. Ex: 0.6"))

            corr_m = []
            if algorithm[0] == 'p':
                corr_m = get_pearson_matrix(hm)
            elif algorithm[0] == 's':
                corr_m = get_spearman_matrix(hm)
            else:
                print("Invalid algorithm.")

            initial_cluster, cluster_matrix = get_similarity_labeling_matrix(corr_m, drug_list, r_cutoff)
            cluster = []
            if algorithm[1] != 'm':
                cluster = group_similar_drugs(initial_cluster, has_outcast_cluster=False)
            else:
                mcl_result, mcl_initial_cluster, _ = get_markov_clusters(cluster_matrix, hm, 2, 2)
                cluster = group_similar_drugs(mcl_initial_cluster, has_outcast_cluster=False)

            # Finally, plot resulting heatmaps
            print("Starting to generate heatmaps, each cluster may take a minute."
                  " You can check the folder while generation takes place.")
            plot_heatmaps_of_wanted_clusters(hm, drug_label_map, cluster, path=folder_name)
            print('Heatmap generation is complete in ', timeit.default_timer() - start)

        elif operation == "comparison_json":
            filename = input("Choose a filename. Ex: pearson_vs_spearman_0.6-r-cutoff")
            filename += ".json"

            r_cutoff = float(input("Choose an r_cutoff. Ex: 0.6"))

            corr_m_pearson = get_pearson_matrix(hm)
            corr_m_spearman = get_spearman_matrix(hm)

            pearson_initial_cluster, pearson_matrix = get_similarity_labeling_matrix(corr_m_pearson, drug_list,
                                                                                     r_cutoff)
            spearman_initial_cluster, spearman_matrix = get_similarity_labeling_matrix(corr_m_spearman, drug_list,
                                                                                       r_cutoff)

            p_vs_s_supercluster_comparison, p_vs_s_supercluster_difference = \
                get_A_vs_B(hm, drug_label_map, pearson_initial_cluster, spearman_initial_cluster)

            print("Starting to generate the json file, should take less than a minute."
                  " Check the file when complete.")
            convert_map_to_json(p_vs_s_supercluster_difference, filename)
            print('Json generation is complete in ', timeit.default_timer() - start)

        elif operation == "graph":
            filename = input("clean_wide_clustering_r=0.6")
            filename += ".html"

            r_cutoff = float(input("Choose an r_cutoff. Ex: 0.6"))

            corr_m_pearson = get_pearson_matrix(hm)
            corr_m_spearman = get_spearman_matrix(hm)

            pearson_initial_cluster, pearson_matrix = get_similarity_labeling_matrix(corr_m_pearson, drug_list,
                                                                                     r_cutoff)
            spearman_initial_cluster, spearman_matrix = get_similarity_labeling_matrix(corr_m_spearman, drug_list,
                                                                                       r_cutoff)

            mcl_result, mcl_initial_cluster, _ = get_markov_clusters(pearson_matrix, hm, 2, 2)

            nt, G_comp = get_comparison_graphs(hm, pearson_matrix, spearman_matrix, mcl_result, isDynamic=False)
            nt.toggle_physics(True)  # False to make it static, not dynamic
            nt.show_buttons(filter_=["nodes", "edges", "physics"])
            nt.repulsion()
            nt.show(filename)

        else:
            print("Please input a valid operation")
