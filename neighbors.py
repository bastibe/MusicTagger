
import os
import numpy as np
import pandas as pd
from collections import Counter

from preprocess_sample_base import preprocess_sample
from extract_features import extract_features
from dynamic_time_warping import dtw_distance_c
    
    
def k_nearest_neighbors(sample_features, k=5):
    ''' returns the tags and the probability of them within 
    the k nearest neighbors
    '''
    sample_dir = 'SampleBase'
    feature_data = pd.read_hdf('feature_data.hdf', 'pca')
    feature_index = np.arange(5)
    neighbors = [(dtw_distance_c(sample_features[feature_index], sample2_features[feature_index]), sample2_features['tag'].iloc[0]) 
        for _, sample2_features in feature_data.groupby('file')]
    neighbors.sort(key = lambda x: x[0])
    neighbors_count = Counter([n[1] for n in neighbors[:k]]).most_common()
    neighbors_count = [(n[0], n[1]/k) for n in neighbors_count]
    return neighbors_count
   
            
if __name__ == '__main__':
    sample_dir = 'Classify/SomeBD.wav'
    # sample_dir = 'Classify/SomeSynth.wav'
    sample_processed_dir = sample_dir[:-4]+'Processed.wav'
    if not os.path.exists(sample_processed_dir):
        preprocess_sample(sample_dir, sample_processed_dir)
    features = extract_features(sample_processed_dir)
    print(k_nearest_neighbors(features, k=20))
    