"""
Tries k-Nearest Neighbor, a supervised learning algorithm.
"""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def run_knn(X_train, y_train, n_neighbors: int = 3):
    """
    Runs k-Nearest Neighbor algorithm with k=n_neighbors.

    :param X_train: The data features to train kNN on.
    :param y_train: The label of the each data observation.
    :param n_neighbors: Number of neighbors to consider.
    :return: knn: When used knn.predict(x_test), returns the predicted label of x_test based on
    the majority of labels of n_neighbors closest neighbors to x_test in X_train.
    """
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)

    return knn


def evaluate_knn(knn: KNeighborsClassifier, X_test, y_test):
    """
    Calculates the accuracy of the knn model.

    :param knn: the kNN model to predict the test.
    :param X_test: The data features to test kNN with.
    :param y_test: The label of the each data observation.
    :return: accuracy: Accuracy of the kNN model.
    """
    num_of_correct = 0
    for i in range(len(X_test)):
        y_pred = knn.predict([X_test[i]])[0]
        if (type(y_pred) == np.str_ and type(y_test[i]) == str) and (y_pred in y_test[i] or y_test[i] in y_pred):
            num_of_correct += 1

    if len(X_test) > 0:
        accuracy = num_of_correct / len(X_test)
    else:
        accuracy = 0

    return accuracy


def run_knn_with_variety_of_k(X_train, y_train, X_val, y_val, k_list: list):
    """
    Tries a variety of k values and returns the accuracy of each kNN model.

    :param X_train: The data features to train kNN on.
    :param y_train: The label of the each data observation of X_train.
    :param X_val: The data features to validate kNN with.
    :param y_val: The label of the each data observation of X_val.
    :param k_list: The list of k values to train kNN models.
    :return:
    """
    acc_list = []
    for k in k_list:
        neigh = run_knn(X_train, y_train, n_neighbors=k)
        acc = evaluate_knn(neigh, X_val, y_val)
        acc_list.append(acc)

    return acc_list
