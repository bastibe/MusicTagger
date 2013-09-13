import pickle
import pandas as pd
import numpy as np
from multiprocessing import Pool
from dynamic_time_warping import dtw_distance_c
from extract_features import extract_features_pca
from docopt import docopt

__doc__ = """
Usage: k_means_classify.py FILE [-d HD5_FILE] [-p PICKLE_FILE]
                                [-c CLASSES] [-b BATCH] [-i ITERATIONS]

Options:
-h --help       show this message
-d HD5_FILE     file name where distances data is stored [default: distances.hd5]
-p PICKLE_FILE  file name where the PCA object is stored [default: pca.pickle]
-c CLASSES      number of classes for k-Means (the k) [default: 10]
-b BATCH        batch size for k-Means [default: 500]
-i ITERATIONS   number of iterations for k-Means [default: 50]
"""

def nearest_class(file, classes, distances):
    """Returns the one class that is closest to the file."""
    return classes[np.argmin(distances[file][classes])]


def find_best_center(cls, files, distances):
    """Selects the one file that has the least sum distance to all other
    files."""
    if not files: return cls
    distances = np.array(distances[files].ix[files])
    return files[np.argmin(np.sum(distances, axis=0))]


def mini_batch_k_means(distances, num_classes, batch_size, num_iterations):
    """Calculates class centroids for a number of classes in a sample base.

    At first, random class centroids are selected in a matrix of
    distances between samples. Then, for each iteration, a batch of
    samples is classified to one of these classes based on its
    distance from the class centroid. Then, for each iteration, the
    class centroids are re-calculated such that the centroid has the
    least distance to all class members.

    Returns the file names of the class centroids.

    """
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


def classify_sample(test_file_name, distances_name, pca_name, num_classes,
                    batch_size, num_iterations):
    """Compares a test file to k-Means class centroids calculated from the
    distances matrix. Returns the class centroid that is closest to
    the test file.

    pca_name is the file name of a pickled PCA object used for
        dimensionality reduction of the test file features.

    num_classes, batch_size and num_iterations are parameters for the
         k-Means algorithm

    """
    distances = pd.read_hdf(distances_name, 'distances')
    with open(pca_name, 'rb') as f: pca = pickle.load(f)
    classes = mini_batch_k_means(distances, num_classes, batch_size,
                                 num_iterations)
    cls_features = [extract_features_pca(cls, pca) for cls in classes]
    file_features = extract_features_pca(test_file_name, pca)
    feature_indices = np.arange(file_features.shape[1]-2)
    cls_dist = np.array([dtw_distance_c(file_features[feature_indices],
                                        cls_feat[feature_indices])
                         for cls_feat in cls_features])
    selected_class = classes[np.argmin(cls_dist)]
    return (selected_class, classes)


if __name__ == '__main__':
    opt = docopt(__doc__)
    selected_class, classes = classify_sample(opt['FILE'], opt['-d'], opt['-p'],
                                              int(opt['-c']), int(opt['-b']),
                                              int(opt['-i']))
    print("class centroids:")
    for cls in classes:
        if cls == selected_class:
            print("--> %s" % cls)
        else:
            print("    %s" % cls)
