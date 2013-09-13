import sys
import os
from pylab import *
sys.path.append('..')

from k_means_classify import *

distances = pd.read_hdf('../distances.hd5', 'distances')
with open('../pca.pickle', 'rb') as f:
    pca = pickle.load(f)

classes = mini_batch_k_means(distances, 10, 500, 50)
classes = ['../'+cls for cls in classes]
cls_features = [extract_features_pca(cls, pca) for cls in classes]

cls2idx = {cls:idx for idx, cls in enumerate(classes)}

sample_dir = '../TestSamples/'
tags = [d for d in os.listdir(sample_dir) if os.path.isdir(sample_dir+d)]
class_histograms = np.zeros((len(tags), len(classes)))
for idx, tag in enumerate(tags):
    tag_dir = tag + '/'
    sample_paths = [sample_dir+tag_dir+sample
                    for sample in os.listdir(sample_dir+tag_dir)
                    if os.path.isfile(sample_dir+tag_dir+sample)]
    feature_indices = np.arange(5)
    for sample in sample_paths:
        sample_features = extract_features_pca(sample, pca)
        cls_distances = np.array([dtw_distance_c(sample_features[feature_indices],
                                                 cls_feat[feature_indices])
                         for cls_feat in cls_features])
        class_histograms[idx, np.argmin(cls_distances)] += 1

imshow(class_histograms, cmap='gray', interpolation='none')
gca().set_xticks(np.arange(10))
gca().set_xticklabels([os.path.basename(os.path.dirname(cls)) for cls in classes], rotation=90)
gca().set_yticks(np.arange(12))
gca().set_yticklabels(tags)
colorbar()
tight_layout()
show()
