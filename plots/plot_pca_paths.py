import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA
from extract_fft import extract_psd
from extract_features import walk_files, extract_features

with open('pca_psd.pickle', 'rb') as f:
    pca_psd = pickle.load(f)
with open('pca_features.pickle', 'rb') as f:
    pca_features = pickle.load(f)

plt.hsv()

feature_names = ['crest_factor', 'log_spectral_centroid', 'peak', 'rms',
                 'spectral_abs_slope_mean', 'spectral_brightness', 'spectral_centroid',
                 'spectral_flatness', 'spectral_skewness', 'spectral_variance']

for root, dirs, files in os.walk('SampleBase'):
    if len(files) <= 1: continue
    fig = plt.figure()
    indices = np.random.random_integers(len(files)-1, size=5)

    ax1 = fig.add_subplot(1,2,1)
    for idx in indices:
        if files[idx].startswith('.'): continue
        psds = extract_psd(root + '/' + files[idx])[np.arange(65)]
        transformed_psds = pca_psd.transform(psds)
        ax1.plot(transformed_psds[:,0], transformed_psds[:,1], '.-.', label=files[idx])
    ax1.legend()
    ax1.set_title(root)

    ax2 = fig.add_subplot(1,2,2)
    for idx in indices:
        if files[idx].startswith('.'): continue
        features = extract_features(root + '/' + files[idx])[feature_names]
        transformed_features = pca_features.transform(features)
        ax2.plot(transformed_features[:,0], transformed_features[:,1], '.-.', label=files[idx])
    ax2.legend()
    ax2.set_title(root)
    plt.show()
