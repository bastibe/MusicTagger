import pickle
import os
import numpy as np
import pandas as pd
from collections import Counter
from extract_features import extract_features_pca
from dynamic_time_warping import dtw_distance_c
from docopt import docopt


__doc__ = """
Usage: knn_classify.py FILE [-k NUM_NEIGHBORS] [-f HD5_FILE] [-p PICKLE_FILE]

Options:
-h --help         show this message
-k NUM_NEIGHBORS  the number of neighbors to consider [default: 10]
-f HD5_FILE       file name where feature data is stored [default: feature_data.hd5]
-p PICKLE_FILE    file name where the PCA object is stored [default: pca.pickle]
"""

def k_nearest_neighbors(sample_features, feature_data, k=5):
    ''' returns the tags and the probability of them within
    the k nearest neighbors
    '''
    neighbors = get_sorted_neighbors(sample_features, feature_data)
    neigh = [(neighbors[idx][0], neighbors[idx][1]) for idx, _ in enumerate(neighbors)]
    neighbors_count = Counter([n[1] for n in neighbors[:k]]).most_common()
    neighbors_count = [(n[0], n[1]/k) for n in neighbors_count]
    return neighbors_count
    
    
def nearest_neighbor(sample_features, feature_data):
    ''' returns the name of the closest sample in the feature 
    space
    '''
    neighbors = get_sorted_neighbors(sample_features, feature_data)
    return neighbors[0][2]
    

def get_sorted_neighbors(sample_features, feature_data):
    ''' creates list with all neighbors and sorts them by the distance
    starting with the lowest
    '''
    feature_index = np.arange(5)
    neighbors = [  (dtw_distance_c(sample_features[feature_index],
                                  sample2_features[feature_index]),
                    sample2_features['tag'].iloc[0],
                    sample2_features['file'].iloc[0])
                 for _, sample2_features in feature_data.groupby('file')]
    neighbors.sort(key = lambda x: x[0])
    return neighbors
    

if __name__ == '__main__':
    opt = docopt(__doc__)
    k = int(opt['-k'])
    test_file_name = opt['FILE']
    features_file_name = opt['-f']
    pca_name = opt['-p']

    with open(pca_name, 'rb') as f: pca = pickle.load(f)
    file_features = extract_features_pca(test_file_name, pca)
    feature_data = pd.read_hdf(features_file_name, 'pca')
    neighbors = k_nearest_neighbors(file_features, feature_data, k=10)

    for name, probability in neighbors:
        print('%s: %i%%' % (name, int(probability*100)))
