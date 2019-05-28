import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif, SelectKBest, SelectPercentile
from sklearn.preprocessing import MinMaxScaler
from skfeature.function.similarity_based import fisher_score
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import time


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


def run_filters(X_train, y_train, X_test, y_test, only_lda=False, only_knn=False):

    nr_features = X_train.shape[1]

    f_scores_filtered_knn = np.zeros(nr_features)
    f_scores_filtered_lda = np.zeros(nr_features)

    mi_scores_filtered_knn = np.zeros(nr_features)
    mi_scores_filtered_lda = np.zeros(nr_features)

    chi2_scores_filtered_knn = np.zeros(nr_features)
    chi2_scores_filtered_lda = np.zeros(nr_features)

    f_score_filter_times = np.zeros(nr_features)
    mi_filter_times = np.zeros(nr_features)
    chi2_filter_times = np.zeros(nr_features)

    f_score_knn_times = np.zeros(nr_features)
    mi_knn_times = np.zeros(nr_features)
    chi2_knn_times = np.zeros(nr_features)

    f_score_lda_times = np.zeros(nr_features)
    mi_lda_times = np.zeros(nr_features)
    chi2_lda_times = np.zeros(nr_features)
    
    f_scores = np.zeros(nr_features)
    mi_scores = np.zeros(nr_features)
    chi2_scores = np.zeros(nr_features)

    for iK, k in enumerate(range(1, nr_features+1)):

        # fscore
        start = time.time()
        fscore = SelectKBest(f_classif, k=k)
        fscore.fit(X_train, y_train)
        end = time.time()
        f_score_filter_times[iK] = end - start
        
        f_knn_filtered_score = 0
        f_lda_filtered_score = 0

        # fscore knn
        if not only_lda:
            start = time.time()
            f_knn_filtered_score = knn(
                fscore.transform(X_train), y_train, fscore.transform(X_test), y_test)
            end = time.time()
            f_score_knn_times[iK] = end - start

        # fscore lda
        if not only_knn:
            start = time.time()
            f_lda_filtered_score = lda(
                fscore.transform(X_train), y_train, fscore.transform(X_test), y_test)
            end = time.time()
            f_score_lda_times[iK] = end - start

        f_scores_filtered_knn[iK] = f_knn_filtered_score
        f_scores_filtered_lda[iK] = f_lda_filtered_score

        # mutual_inf
        start = time.time()
        mi_score = SelectKBest(f_classif, k=k)
        mi_score.fit(X_train, y_train)
        end = time.time()
        mi_filter_times[iK] = end - start
        mi_knn_filtered_score = 0
        mi_lda_filtered_score = 0

        # mutal information knn
        if not only_lda:
            start = time.time()
            mi_knn_filtered_score = knn(
                mi_score.transform(X_train), y_train, mi_score.transform(X_test), y_test)
            end = time.time()
            mi_knn_times[iK] = end - start

        # mutal information lda
        if not only_knn:
            start = time.time()
            mi_lda_filtered_score = lda(
                mi_score.transform(X_train), y_train, mi_score.transform(X_test), y_test)
            end = time.time()
            mi_lda_times[iK] = end - start

            mi_scores_filtered_knn[iK] = mi_knn_filtered_score
            mi_scores_filtered_lda[iK] = mi_lda_filtered_score

        # Chi2
        scaler = MinMaxScaler()
        scaler.fit(X_train, y_train)
        # We need [0,1] for all values for chi2 to work
        X_train_normalized = scaler.transform(X_train)
        X_test_normalized = scaler.transform(X_test)
        start = time.time()
        X_chi2 = SelectKBest(chi2, k=k)
        X_chi2.fit(X_train_normalized, y_train)
        end = time.time()
        chi2_filter_times[iK] = end - start

        X_train_chi2 = X_chi2.transform(X_train_normalized)
        X_test_chi2 = X_chi2.transform(X_test_normalized)
        chi2_knn_filtered_score = 0
        chi2_lda_filtered_score = 0

        # Chi2 knn
        if not only_lda:
            start = time.time()
            chi2_knn_filtered_score = knn(
                X_train_chi2, y_train, X_test_chi2, y_test)
            end = time.time()
            chi2_knn_times[iK] = end - start

        # Chi2 lda
        if not only_knn:
            start = time.time()
            chi2_lda_filtered_score = lda(
                X_train_chi2, y_train, X_test_chi2, y_test)
            end = time.time()
            chi2_lda_times[iK] = end - start

            chi2_scores_filtered_knn[iK] = chi2_knn_filtered_score
            chi2_scores_filtered_lda[iK] = chi2_lda_filtered_score

    #print("Order: f_score knn, mutual information knn, chi2 knn, f score lda, mutual information lda, chi2 lda")

    best_ks = np.argmax(f_scores_filtered_knn)+1, np.argmax(mi_scores_filtered_knn)+1, np.argmax(
        chi2_scores_filtered_knn)+1, np.argmax(f_scores_filtered_lda)+1, np.argmax(mi_scores_filtered_lda)+1, np.argmax(chi2_scores_filtered_lda)+1

    best_scores = np.max(f_scores_filtered_knn), np.max(mi_scores_filtered_knn), np.max(
        chi2_scores_filtered_knn), np.max(f_scores_filtered_lda), np.max(mi_scores_filtered_lda), np.max(chi2_scores_filtered_lda)

    times = np.sum(f_score_knn_times) + np.sum(f_score_filter_times), np.sum(mi_knn_times) + np.sum(mi_filter_times), np.sum(
        chi2_knn_times) + np.sum(chi2_filter_times), np.sum(f_score_lda_times)+np.sum(f_score_filter_times), np.sum(mi_lda_times) + np.sum(mi_filter_times), np.sum(chi2_lda_times) + np.sum(chi2_filter_times)

    return best_ks, best_scores, times, f_scores_filtered_knn, f_scores_filtered_lda, mi_scores_filtered_knn, mi_scores_filtered_lda, chi2_scores_filtered_knn, chi2_scores_filtered_lda
