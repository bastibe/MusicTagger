import sys
import os
import pickle
from pylab import *
import pandas as pd
import pdb

sys.path.append('..')

from knn_classify import nearest_neighbor
from extract_features import walk_files, extract_features_pca


feature_data = pd.read_hdf('../feature_data.hd5', 'pca')
with open('../pca.pickle', 'rb') as f:
    pca = pickle.load(f)


sample_dir = '../TestSamples/'
tags = [d for d in os.listdir(sample_dir) if os.path.isdir(sample_dir+d)]
for idx, tag in enumerate(tags):
    tag_dir = tag + '/'
    sample_paths = [sample_dir+tag_dir+sample
                    for sample in os.listdir(sample_dir+tag_dir)
                    if os.path.isfile(sample_dir+tag_dir+sample)]
    print(tag, idx)
    picked_samples = {}
    for test_file in sample_paths:
        file_feature = extract_features_pca(test_file, pca)
        picked_samples[test_file] = nearest_neighbor(file_feature, feature_data)
    print(picked_samples)