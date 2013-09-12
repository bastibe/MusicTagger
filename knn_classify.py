import os
import numpy as np
import pandas as pd
from collections import Counter
from preprocess_sample_base import preprocess_sample
from extract_features import extract_features
from dynamic_time_warping import dtw_distance_c


def k_nearest_neighbors(sample_features, feature_data, k=5):
    ''' returns the tags and the probability of them within
    the k nearest neighbors
    '''
    feature_index = np.arange(5)
    neighbors = [(dtw_distance_c(sample_features[feature_index], sample2_features[feature_index]), sample2_features['tag'].iloc[0])
        for _, sample2_features in feature_data.groupby('file')]
    neighbors.sort(key = lambda x: x[0])
    neighbors_count = Counter([n[1] for n in neighbors[:k]]).most_common()
    neighbors_count = [(n[0], n[1]/k) for n in neighbors_count]
    return neighbors_count


if __name__ == '__main__':
    test_file_name = sys.argv[1]
    features_file_name = sys.argv[2] if len(sys.argv) > 2 else 'feature_data.hd5'
    pca_name = sys.argv[3] if len(sys.argv) > 3 else 'pca.pickle'

    with open(pca_name, 'rb'): pca = pickle.read()
    file_features = extract_features_pca(test_file_name, pca)
    features_data = pd.read_hdf(features_file_name, 'pca')
    print(k_nearest_neighbors(file_features, feature_data, k=20))
