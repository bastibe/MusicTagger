import pandas as pd
import numpy as np
from multiprocessing import Pool
from dynamic_time_warping import dtw_distance_c

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
    test_file_name = sys.argv[1]
    distances_name = sys.argv[2] if len(sys.argv) > 2 else 'distances.hd5'
    pca_name = sys.argv[3] if len(sys.argv) > 3 else 'pca.pickle'

    distances = pd.read_hdf(distances_name, 'distances')
    with open(pca_name, 'rb'): pca = pickle.read()
    classes = mini_batch_k_means(distances, 10, 500, 50)
    cls_features = [extract_features_pca(cls, pca) for cls in classes]
    file_features = extract_features_pca(test_file_name, pca)
    cls_dist = np.array([dtw_distance_c(file_features, cls_feat) for cls_feat in cls_features])
    return classes[np.argmin(cls_dist)]
