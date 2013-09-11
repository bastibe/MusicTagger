import pickle
import pandas as pd
import numpy as np
from multiprocessing import Pool
from dynamic_time_warping import dtw_distance_c
from operator import iadd
from functools import reduce


def compare_file(data1, all_data):
    """Compares the features in data1 to the features in each file in
    all_data and returns a list of ('name', 'name', distance) tuples.

    """
    file1 = data1['file'].iloc[0]
    feature_idx = np.arange(5)
    distances = [(file1, file2, dtw_distance_c(data1[feature_idx],
                                               data2[feature_idx]))
                 for file2, data2 in all_data.groupby('file')]
    return distances


def compare_file_to_hdf(data):

    """Compares one file to all files and returns a list.
    This is a helper function for the multiprocessing.Pool.

    """
    all_data = pd.read_hdf('feature_data.hdf', 'pca')
    return compare_file(data, all_data)


def compare_save_all_data():
    """Compares all files to all files, then saves the resulting matrix.
    This actually calculates the distances between all pairs, so both
    x <-> x and x <-> y and y <-> x. Computation could be sped up by
    removing these.

    As it is, this will take about 1:40 h to complete on a 4-core Unix
    machine. Do not try this on Windows.

    """
    feature_data = pd.read_hdf('feature_data.hdf', 'pca')
    pool = Pool(processes=4)
    arguments = [data for _, data in feature_data.groupby('file')]
    # this produces a list of lists of ('file', 'file', distance) tuples
    distances = pool.map(compare_file_to_hdf, arguments, chunksize=100)
    distances = reduce(iadd, distances) # concatenate all sub-lists
    files = feature_data['file'].unique()
    file_map = {file:idx for idx, file in enumerate(files)}
    distance_mat = np.zeros((len(files), len(files)))
    for file1, file2, distance in distances:
        distance_mat[file_map[file1], file_map[file2]] = distance
    distances_df = pd.DataFrame(distance_mat, index=files, columns=files)
    distances_df.to_hdf('distances.hdf', 'distances')


def nearest_class(file, classes, distances):
    return classes[np.argmin(distances[file][classes])]


def find_best_center(cls, files, distances):
    if not files: return cls
    distances = np.array(distances[files].ix[files])
    return files[np.argmin(np.sum(distances, axis=0))]


def mini_batch_k_means(distances, num_classes, batch_size, num_iterations):
    files = np.array(distances.columns)
    classes = {f:[] for f in files[np.random.randint(len(files), size=num_classes)]}
    for iteration in range(num_iterations):
        examples = files[np.random.randint(len(files), size=batch_size)]
        for ex in examples:
            cls = nearest_class(ex, list(classes.keys()), distances)
            classes[cls] += [ex]
        classes = {find_best_center(cls, examples, distances):[]
                   for cls, examples in classes.items()}
    return list(classes.keys())


if __name__ == '__main__':
    distances = pd.read_hdf('distances.hdf', 'distances')
    classes = mini_batch_k_means(distances, 10, 500, 50)
