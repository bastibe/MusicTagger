import pickle
import pandas as pd
from sklearn.decomposition import PCA


"""Reads the entry 'features' from 'feature_data.hdf', calculates the
five most significant principal components, and writes them into the
entry 'pca' in the same file.

"""


def calculate_pca(feature_data, num_components):
    """Create a PCA object for a dataset"""
    pca = PCA(n_components=num_components)
    pca.fit(feature_data)
    return pca


if __name__ == "__main__":
    feature_data = pd.read_hdf('feature_data.hdf', 'features')
    pca = calculate_pca(feature_data[list(range(10))], 5)

    pca_data = feature_data[['tag', 'file']]
    features = pca.transform(feature_data[list(range(10))])
    for n in range(5):
        pca_data.insert(n, n, features[:,n])
    pca_data.to_hdf('feature_data.hdf', 'pca')
