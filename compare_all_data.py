import sys
import numpy as np
import pandas as pd
from multiprocessing import Pool
from dynamic_time_warping import dtw_distance_c
from operator import iadd
from functools import reduce
from docopt import docopt


__doc__ = \
"""Usage: compare_all_data.py [-h] [-f HD5_FILE] [-d HD5_FILE]

Warning: This will take about 1:30 h on a recent four-core Unix machine
         On Window it will take 6 h or more, since multiprocessing is not available

Options:
-h --help       show this
-f HD5_FILE     file name where feature data is stored [default: feature_data.hd5]
-d HD5_FILE     file name where distances should be stored [default: distances.hd5]
"""


def compare_file(data1, all_data):
    """Compares the features in data1 to the features in each file in
    all_data and returns a list of ('name', 'name', distance) tuples.

    """
    file1 = data1['file'].iloc[0]
    feature_indices = np.arange(data1.shape[1]-2)
    distances = [(file1, file2, dtw_distance_c(data1[feature_indices],
                                               data2[feature_indices]))
                 for file2, data2 in all_data.groupby('file')]
    return distances


def compare_file_to_hdf_helper(data):
    """Compares one file to all files and returns a list.
    This is a helper function for the multiprocessing.Pool.

    """
    return compare_file(data[0], data[1])


def compare_all_samples(feature_data):
    """Compares all files to all files, then saves the resulting matrix.
    This actually calculates the distances between all pairs, so both
    x <-> x and x <-> y and y <-> x. Computation could be sped up by
    removing these.

    As it is, this will take about 1:40 h to complete on a 4-core Unix
    machine. Takes four times as long on Windows.

    """
    arguments = [(data, feature_data) for _, data in feature_data.groupby('file')]
    # this produces a list of lists of ('file', 'file', distance) tuples
    if sys.platform == 'win32':
        distances = map(compare_file_to_hdf_helper, arguments)
    else:
        pool = Pool(processes=4)
        distances = pool.map(compare_file_to_hdf_helper, arguments, chunksize=100)
    distances = reduce(iadd, distances) # concatenate all sub-lists
    files = feature_data['file'].unique()
    file_map = {file:idx for idx, file in enumerate(files)}
    distance_mat = np.zeros((len(files), len(files)))
    for file1, file2, distance in distances:
        distance_mat[file_map[file1], file_map[file2]] = distance
    distances_df = pd.DataFrame(distance_mat, index=files, columns=files)
    return distances_df


if __name__ == '__main__':
    options = docopt(__doc__)
    features_name = options['-f']
    distances_name = options['-d']
    feature_data = pd.read_hdf(features_name, 'pca')
    distances = compare_all_samples(feature_data)
    distances.to_hdf(distances_name, 'distances')
