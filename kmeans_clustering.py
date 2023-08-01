from sklearn.cluster import KMeans

def run_kmeans(X_train, k=2):
    """
    Run k-Means with k=k.

    :param X_train: The data features to train k-Means on.
    :param k: Number of clusters wanted.
    :return: When used kmeans.predict(x_test), returns the predicted label of x_test based on
    the closest centroid to it in the kmeans model.
    """
    kmeans = KMeans(n_clusters=k, random_state=6).fit(X_train)
    return kmeans



def cluster_drugs(kmeans: KMeans, k, X_test, hm):
    """
    kmeans.predict(X_test) returns a list of integers, where each integer shows which cluster
    the drug in that index belongs to. This function sees each integer as a group, and clusters
    each drug that has the same integer.
    For example, kmeans.predict(X_test) might return [0, 0, 1], meaning that the first and the
    second drug is clustered together, whereas the third is not. This function finds the names
    of these drugs and puts the first and the second drug in the same list

    :param kmeans: The k-Means model that is already fit.
    :param k: The number of clusters that the k-Means model was fit
    :param X_test: The data features to test kmeans with.
    :param hm: Heatmap (DataFrame) of drugs, to find the name of the drugs
    :return: List of clusters, where each cluster is a list of drugs (strings)
    """
    clusters = []
    for i in range(k):
        clusters.append([])

    for i in range(len(X_test)):
        y_class = kmeans.predict([X_test[i]])[0]
        drug = '?'
        for j in range(len(hm.values)):
            if (hm.values[j] == X_test[i]).all():  # ignore the red highlight, it works
                drug = hm.index[j]
                break
        clusters[y_class].append(drug)

    return clusters


def run_kmeans_with_variety_of_k(hm, X_train, X_val, k_list):
    """
    Tries a variety of k values and returns the list of clusters found by each model. Then, this
    list can be used to compute the Normlized Mutual Information score of each k-Means model.
    Since this approach is unsupervised, we do not calculate the accuracy, but use other
    evaluation metrics.

    :param X_train: The data features to train k-Means on.
    :param X_val: The data features to validate k-Means with.
    :param k_list: The list of k values to train kNN models.
    :return:
    """
    cluster_list = []
    for k in k_list:
        kmeans = KMeans(n_clusters=k, random_state=6).fit(X_train)
        cluster = cluster_drugs(kmeans, k, X_val, hm)  # ignore the red highlight, it works
        while [] in cluster:
                cluster.remove([])  # removes empty clusters to not run into errors with NMI

        cluster_list.append(cluster)

    return cluster_list
