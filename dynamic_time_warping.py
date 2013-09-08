import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

def distance_matrix(path1, path2):
    """Calculate the euclidean distance between every combination of
    feature vectors in the two paths."""
    distances = np.zeros((path1.shape[0], path2.shape[0]))
    for row in range(distances.shape[0]):
        for col in range(distances.shape[1]):
            distances[row, col] = np.sqrt(np.sum((path1.iloc[row]-path2.iloc[col])**2))
    return distances

def extract_path(path_matrix):
    """Given a matrix that contains cheapest path vectors in the form:
    cheapest is left: 1
    cheapest is down: 2
    cheapest is down and left: 3
    Find the cheapest path from (0,0) to (max,max).
    """
    optimal_path_matrix = np.zeros(path_matrix.shape)
    [row, col] = [x-1 for x in path_matrix.shape]
    while row != 0 and col != 0:
        optimal_path_matrix[row, col] = 1
        if path_matrix[row, col] == 1:
            col -= 1
        elif path_matrix[row, col] == 2:
            row -= 1
        else:
            row -= 1
            col -= 1
    optimal_path_matrix[0,0] = 1
    return np.where(optimal_path_matrix == 1)

def search_optimal_path_verbose(costs):
    """Given a matrix of cost values, calculate the path through this
    matrix with the smallest cumulative cost.

    Also, return the cumulative cost matrix and the optimal path
    through this matrix.

    """
    cumulative_costs = np.zeros(costs.shape)
    path = np.zeros(costs.shape)
    for row in range(costs.shape[0]):
        for col in range(costs.shape[1]):
            if col == 0 and row == 0:
                cumulative_costs[row, col] = costs[row, col]
                path[row, col] = 0
            elif row == 0:
                cumulative_costs[row, col] = (cumulative_costs[row, col-1] +
                                              costs[row, col])
                path[row, col] = 1
            elif col == 0:
                cumulative_costs[row, col] = (cumulative_costs[row-1, col] +
                                              costs[row, col])
                path[row, col] = 2
            else:
                horizontal = cumulative_costs[row, col-1] + costs[row, col]
                vertical   = cumulative_costs[row-1, col] + costs[row, col]
                diagonal   = cumulative_costs[row-1, col-1] + 2*costs[row, col]
                cumulative_costs[row, col] = np.min((horizontal, vertical, diagonal))
                path[row, col] = np.argmin((horizontal, vertical, diagonal))+1
    min_cost = cumulative_costs[-1,-1]
    min_path = extract_path(path)
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(costs, cmap=cm.gray)
    plt.title("Cost Matrix")
    ax = plt.subplot(1,2,2)
    plt.imshow(cumulative_costs, cmap=cm.gray)
    plt.title("Cumulative Cost and Cheapest Path")
    xlim, ylim = (ax.get_xlim(), ax.get_ylim())
    plt.plot(*min_path[::-1], color='red')
    plt.gca().set_xlim(xlim)
    plt.gca().set_ylim(ylim)
    plt.show()
    return min_cost

def search_optimal_path(costs):
    """Given a matrix of cost values, calculate the path through this
    matrix with the smallest cumulative cost."""
    cumulative_costs = np.zeros(costs.shape)
    for row in range(costs.shape[0]):
        for col in range(costs.shape[1]):
            if col == 0 and row == 0:
                cumulative_costs[row, col] = costs[row, col]
            elif row == 0:
                cumulative_costs[row, col] = (cumulative_costs[row, col-1] +
                                              costs[row, col])
            elif col == 0:
                cumulative_costs[row, col] = (cumulative_costs[row-1, col] +
                                              costs[row, col])
            else:
                horizontal = cumulative_costs[row, col-1] + costs[row, col]
                vertical   = cumulative_costs[row-1, col] + costs[row, col]
                diagonal   = cumulative_costs[row-1, col-1] + 2*costs[row, col]
                cumulative_costs[row, col] = np.min((horizontal, vertical, diagonal))
    return cumulative_costs[-1,-1]


def dtw_distance(path1, path2):
    """Calculate distance between two feature space paths by
    time-stretching both feature space paths for maximum
    similarity.

    """
    distances = distance_matrix(path1, path2)
    min_distance = search_optimal_path(distances)
    return min_distance / np.sum(distances.shape)


if __name__ == '__main__':
    features = ['crest_factor', 'log_spectral_centroid', 'peak', 'rms',
            'spectral_abs_slope_mean', 'spectral_brightness', 'spectral_centroid',
            'spectral_flatness', 'spectral_skewness', 'spectral_variance']
    feature_data = pd.read_hdf('data.hdf', 'features')
    first_tag = feature_data['tag'].unique()[0]
    tagged_files = feature_data[feature_data['tag'] == first_tag]['file'].unique()

    first_file = feature_data[feature_data['file'] == tagged_files[0]]
    second_file = feature_data[feature_data['file'] == tagged_files[1]]

    print("The distance between %s (%i blocks) and %s (%i blocks) is:" %
          (tagged_files[0], first_file[features].shape[0],
           tagged_files[1], second_file[features].shape[0]), end='')
    print(" %f" % dtw_distance(first_file[features], second_file[features]))