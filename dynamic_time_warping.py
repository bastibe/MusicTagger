import pickle
import sys
import numpy as np
import pandas as pd
from cffi import FFI
import matplotlib.pyplot as plt
from matplotlib import cm
from extract_features import extract_features_pca
from docopt import docopt


__doc__ = """Usage: dynamic_time_warping.py FILE1 FILE2 [-p PICKLE_FILE]

Options:
-h --help       this message
-p PICKLE_FILE  the file containing the pickled PCA object [default: pca.pickle]
"""


ffi = FFI()
ffi.cdef('void distance_matrix(float*, int, float*, int, int, float*);')
ffi.cdef('float search_optimal_path(float*, float*, int, int);')
if sys.platform == 'win32':
    _c = ffi.dlopen('dtw.dll')
else:
    _c = ffi.dlopen('dtw.so')

def distance_matrix(path1, path2):
    """Calculate the euclidean distance between every combination of
    feature vectors in the two paths."""
    distances = np.zeros((path1.shape[0], path2.shape[0]))
    for row in range(distances.shape[0]):
        for col in range(distances.shape[1]):
            distances[row, col] = np.sqrt(np.sum((path1[row:row+1]-path2[col:col+1])**2))
    return distances


def search_optimal_path(costs):
    """Given a matrix of cost values, calculate the path through this
    matrix with the smallest cumulative cost."""
    cumulative_costs = np.zeros(costs.shape)
    cost_vert = 1/costs.shape[0]
    cost_horz = 1/costs.shape[1]
    cost_diag = np.sqrt(cost_vert**2 + cost_horz**2)
    for row in range(costs.shape[0]):
        for col in range(costs.shape[1]):
            if col == 0 and row == 0:
                cumulative_costs[row, col] = cost_diag*costs[row, col]
            elif row == 0:
                cumulative_costs[row, col] = (cumulative_costs[row, col-1] +
                                              cost_horz*costs[row, col])
            elif col == 0:
                cumulative_costs[row, col] = (cumulative_costs[row-1, col] +
                                              cost_vert*costs[row, col])
            else:
                horizontal = cumulative_costs[row, col-1] + cost_horz*costs[row, col]
                vertical   = cumulative_costs[row-1, col] + cost_vert*costs[row, col]
                diagonal   = cumulative_costs[row-1, col-1] + cost_diag*costs[row, col]
                cumulative_costs[row, col] = np.min((horizontal, vertical, diagonal))
    return cumulative_costs[-1,-1]


def dtw_distance(path1, path2):
    """Calculate distance between two feature space paths by
    time-stretching both feature space paths for maximum
    similarity.

    """
    distances = distance_matrix(path1, path2)
    min_distance = search_optimal_path(distances)
    return min_distance


def dtw_distance_c(path1, path2):
    """Calculate distance between two feature space paths by
    time-stretching both feature space paths for maximum
    similarity.

    """
    num_distances =  path1.shape[0]*path2.shape[0]
    path1_c = ffi.new('char[]', np.array(path1, dtype=np.float32).tostring())
    path2_c = ffi.new('char[]', np.array(path2, dtype=np.float32).tostring())
    distances_c = ffi.new('float[]', num_distances)
    _c.distance_matrix(ffi.cast('float*', path1_c), path1.shape[0],
                       ffi.cast('float*', path2_c), path2.shape[0],
                       path1.shape[1], distances_c)
    cumulative_costs_c = ffi.new('float[]', num_distances)
    min_distance = _c.search_optimal_path(distances_c, cumulative_costs_c, path1.shape[0], path2.shape[0])
    return min_distance


if __name__ == '__main__':
    options = docopt(__doc__)
    with open(options['-p'], 'rb') as f:
        pca = pickle.load(f)
    first_file = extract_features_pca(options['FILE1'], pca)
    second_file = extract_features_pca(options['FILE2'], pca)
    feature_cols = np.arange(first_file.shape[1]-2)

    print("The distance between %s and %s is: %f" %
          (options['FILE1'], options['FILE2'],
           dtw_distance_c(first_file[feature_cols], second_file[feature_cols])))
