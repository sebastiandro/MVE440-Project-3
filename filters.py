import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif, SelectKBest, SelectPercentile
from sklearn.preprocessing import MinMaxScaler
from skfeature.function.similarity_based import fisher_score
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def lda(x_filtered_train, y_filtered_train, x_filtered_test, y_filtered_test):

    lda_filtered_score = -1
    lda_unfiltered_score = -1
    # LDA classifier filtered data
    lda = LinearDiscriminantAnalysis()
    lda.fit(x_filtered_train, y_filtered_train)
    lda_filtered_score = lda.score(x_filtered_test, y_filtered_test)

    return lda_filtered_score


def knn(x_filtered_train, y_filtered_train, x_filtered_test, y_filtered_test):

    knn_filtered_score = -1
    knn_unfiltered_score = -1

    # Pruned data classifier
    neigh = KNeighborsClassifier()
    neigh.fit(x_filtered_train, y_filtered_train)
    knn_filtered_score = neigh.score(x_filtered_test, y_filtered_test)

    return knn_filtered_score


def run_filters(X_train, y_train, X_test, y_test):

    nr_features = X_train.shape[1]

    f_scores_filtered_knn = np.zeros(nr_features)
    f_scores_filtered_lda = np.zeros(nr_features)

    mi_scores_filtered_knn = np.zeros(nr_features)
    mi_scores_filtered_lda = np.zeros(nr_features)

    chi2_scores_filtered_knn = np.zeros(nr_features)
    chi2_scores_filtered_lda = np.zeros(nr_features)

    for iK, k in enumerate(range(1, nr_features)):

        # fscore
        fscore = SelectKBest(f_classif, k=k)
        fscore.fit(X_train, y_train)

        f_knn_filtered_score = knn(
            fscore.transform(X_train), y_train, fscore.transform(X_test), y_test)

        f_lda_filtered_score = lda(
            fscore.transform(X_train), y_train, fscore.transform(X_test), y_test)

        f_scores_filtered_knn[iK] = f_knn_filtered_score
        f_scores_filtered_lda[iK] = f_lda_filtered_score

        # mutual_inf
        mi_score = SelectKBest(f_classif, k=k)
        mi_score.fit(X_train, y_train)

        mi_knn_filtered_score = knn(
            mi_score.transform(X_train), y_train, mi_score.transform(X_test), y_test)

        mi_lda_filtered_score = lda(
            mi_score.transform(X_train), y_train, mi_score.transform(X_test), y_test)

        mi_scores_filtered_knn[iK] = mi_knn_filtered_score
        mi_scores_filtered_lda[iK] = mi_lda_filtered_score

        # Chi2
        scaler = MinMaxScaler()
        scaler.fit(X_train, y_train)
        # We need [0,1] for all values for chi2 to work
        X_train_normalized = scaler.transform(X_train)
        X_test_normalized = scaler.transform(X_test)
        X_chi2 = SelectKBest(chi2, k=k)
        X_chi2.fit(X_train_normalized, y_train)

        X_train_chi2 = X_chi2.transform(X_train_normalized)
        X_test_chi2 = X_chi2.transform(X_test_normalized)

        chi2_knn_filtered_score = knn(
            X_train_chi2, y_train, X_test_chi2, y_test)
        chi2_lda_filtered_score = lda(
            X_train_chi2, y_train, X_test_chi2, y_test)

        chi2_scores_filtered_knn[iK] = chi2_knn_filtered_score
        chi2_scores_filtered_lda[iK] = chi2_lda_filtered_score

    print("Order: f_score knn, f score lda, mutual information knn, mutual information lda, chi2 knn, chi2 lda")

    best_ks = np.argmax(f_scores_filtered_knn)+1, np.argmax(f_scores_filtered_lda)+1, np.argmax(mi_scores_filtered_knn)+1, np.argmax(mi_scores_filtered_lda)+1, np.argmax(
        chi2_scores_filtered_knn)+1, np.argmax(chi2_scores_filtered_lda)+1

    best_scores = np.max(f_scores_filtered_knn), np.max(f_scores_filtered_lda), np.max(mi_scores_filtered_knn), np.max(mi_scores_filtered_lda), np.max(
        chi2_scores_filtered_knn), np.max(chi2_scores_filtered_lda)

    return best_ks, best_scores
