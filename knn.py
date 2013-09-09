
import numpy as np
import pandas as pd
from sklearn import datasets
# from extract_features import walk_files


digit_data = datasets.load_digits()


def get_distance(datapoint, sample):
    distance = datapoint-digit_data['data'][sample:sample+1]
    distance *= distance
    distance = np.sqrt(np.sum(distance))
    return distance, digit_data['target'][sample]
    
    
def knn(k=5):
    datapoint = np.random.rand(1,64)*16
    neighbors = {'distance': [], 'tag': []}
    # for sample in walk_files(sample_path):
    for sample in range(1797):
        (distance, tag) = get_distance(datapoint, sample)
        neighbors['distance'].append(distance)
        neighbors['tag'].append(tag)
    neighbors = pd.DataFrame(neighbors)
    neighbors = neighbors.sort(columns='distance')
    nearest_neighbors = pd.value_counts(neighbors['tag'][:k])
    nearest_neighbor = {'tag': nearest_neighbors.index[0], 'count':  nearest_neighbors.iget(0)}
    return nearest_neighbor
            
            
if __name__ == '__main__':
    print(knn(50))