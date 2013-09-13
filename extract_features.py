import pickle
import os
import sys
import inspect
import pandas as pd
from multiprocessing import Pool
from scipy.signal import hann
from numpy.fft import rfft
from pysoundfile import SoundFile, read_mode
from preprocess import preprocess_sample_data
from sklearn.decomposition import PCA
import features
from docopt import docopt

__doc__ = \
"""Usage: extract_features.py [-h] [-s DIR] [-f HD5_FILE] [-p PICKLE_FILE] [--pca=K]

Options:
-h --help       show this
-s DIR          directory where all samples are stored [default: Samples]
-f HD5_FILE     file name where feature data should be saved [default: feature_data.hd5]
-p PICKLE_FILE  file name where the PCA object should be saved [default: pca.pickle]
--pca=K         number of features after PCA [default: 5]
"""


def all_features():
    """Return all functions defined in the module "features"."""
    for name, function in inspect.getmembers(features):
        if inspect.isfunction(function):
            yield(name, function)


def walk_files(sample_path):
    """Iterate over all files in subdirectories of sample_path.

    Dotfiles are ignored.

    """
    for root, dirs, files in os.walk(sample_path):
        for idx, file in enumerate(files):
            if file.startswith('.'): continue
            yield(root+'/'+file)


def blocks(samples, block_len, overlap=0.5, window=hann):
    """Returns blocks of audio data from a sound file.

    Each block will be of length block_len. The last block in the file
    is likely to be shorter. Blocks will overlap according to overlap
    and be windowed by the window function. window is a function that
    takes a numeric argument and returns a window of that length.

    This will only read the first channel if there are more than one.

    """
    read_position = int(block_len*overlap)
    while read_position < len(samples)-1:
        read_position -= int(block_len*overlap)
        data = samples[read_position:read_position+block_len]
        read_position += block_len
        data *= window(len(data))
        yield(data, rfft(data))


def extract_features(path, block_len_sec=0.02):
    """Calculates features for each block of a sound file at a given path.

    This reads the sound file at path, cuts it up in windowed,
    overlapping blocks of length block_len_sec (in seconds), and
    executes all functions defined in the module "features" on all
    blocks.

    Returns a pandas DataFrame with feature names as columns and block
    indices as rows.

    """
    file = SoundFile(path, mode=read_mode)
    block_len = int(file.sample_rate*block_len_sec)
    samples = preprocess_sample_data(file[:])
    feature_data = { 'file': os.path.relpath(path),
                     'tag': os.path.basename(os.path.dirname(path)) }
    for name, func in all_features():
        feature_data[name] = [func(*data) for data in blocks(samples, block_len)]
    return pd.DataFrame(feature_data, columns=([name for name, _ in all_features()] + ['file', 'tag']))


def calculate_pca(feature_data, num_components):
    """Create a PCA object for a dataset"""
    pca = PCA(n_components=num_components)
    pca.fit(feature_data)
    return pca


def extract_features_pca(path, pca, block_len_sec=0.02):
    features = extract_features(path, block_len_sec)
    meta_data = features[['tag', 'file']]
    feature_indices = np.arange(features.shape[1]-2)
    feature_data = features[feature_indices]
    pca_data = pca.transform(feature_data)
    pca_feature_data = meta_data
    for n in range(pca_data.shape[1]):
        meta_data.insert(n, n, pca_data[:,n])
    return meta_data


if __name__ == '__main__':
    options = docopt(__doc__)
    sample_path = options['-s']
    hdf_name = options['-f']
    pca_name = options['-p']
    pca_count = int(options['--pca'])

    # calculate feature data
    if sys.platform == 'win32':
        feature_data = pd.concat([extract_features(f) for f in walk_files(sample_path)])
    else:
        pool = Pool(processes=4)
        feature_data = pd.concat(pool.map(extract_features, walk_files(sample_path), chunksize=100))
    feature_data.to_hdf(hdf_name, 'features')

    # calculate principal component analysis
    feature_indices = np.arange(feature_data.shape[1]-2)
    pca = calculate_pca(feature_data[feature_indices], pca_count)
    with open(pca_name, 'wb') as f: pickle.dump(pca, f)

    # calculate reduced feature data
    pca_features = pca.transform(feature_data[feature_indices])
    feature_data = feature_data[['tag', 'file']]
    for n in range(pca_count):
        feature_data.insert(n, n, pca_features[:,n])
    feature_data.to_hdf(hdf_name, 'pca')
