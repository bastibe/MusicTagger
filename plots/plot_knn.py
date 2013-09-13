import sys
import os
import pickle
from collections import Counter
from pylab import *
#import numpy as np
import pandas as pd
sys.path.append('..')

from knn_classify import k_nearest_neighbors
from extract_features import walk_files, extract_features_pca


feature_data = pd.read_hdf('../feature_data.hd5', 'pca')
with open('../pca.pickle', 'rb') as f:
    pca = pickle.load(f)

    
sample_dir = '../TestSamples/'
tags = [d for d in os.listdir(sample_dir) if os.path.isdir(sample_dir+d)]
all_classifications = {}
for t in tags:
    all_classifications[t] = []
count_all = []
for idx, tag in enumerate(tags):
    tag_dir = tag + '/'
    #if idx > 1:
    #    continue
    sample_paths = [sample_dir+tag_dir+sample
                    for sample in os.listdir(sample_dir+tag_dir)
                    if os.path.isfile(sample_dir+tag_dir+sample)]
    print(tag, idx)
    for test_file in sample_paths:
        file_feature = extract_features_pca(test_file, pca)
        all_classifications[tag] += [k_nearest_neighbors(file_feature, feature_data, k=10)[0]]
    count_all += [Counter([b[0] for b in all_classifications[tag]]).most_common()]

    
tag2idx = {tag:idx for idx, tag in enumerate(tags)}
knn_histogram = zeros((len(tags), len(tags)))
for idx, item in enumerate(count_all):
    for tag, count in item:
        knn_histogram[idx, tag2idx[tag]] = count
        
        
ax = imshow(knn_histogram, cmap='gray', interpolation='none')
ax.axes.set_xticks(np.arange(12))
ax.axes.set_xticklabels(tags, rotation=90)
ax.axes.set_yticks(np.arange(12))
ax.axes.set_yticklabels(tags)
colorbar()
ax.axes.set_position((-0.3,0.3,1,0.6))
gcf().savefig('knn.png')