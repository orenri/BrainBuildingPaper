import numpy as np
import pickle
import os
from er_block_model import generate_er_block_per_type
from CElegansNeuronsAdder import CElegansNeuronsAdder


def train_types_and_distances_model(beta, type_configuration, data_path, neurons_subset_path,
                                    data_dir_path=CElegansNeuronsAdder.DEFAULT_DATA_PATH, num_artificial_types=1,
                                    types_file_name=CElegansNeuronsAdder.DEFAULT_TYPES_FILE_NAME):
    _, er_block_average = generate_er_block_per_type(type_configuration, data_path, neurons_subset_path,
                                                     c_elegans_data_path=data_dir_path,
                                                     num_artificial_types=num_artificial_types,
                                                     types_file_name=types_file_name)

    with open(os.path.join(data_dir_path, data_path), 'rb') as f:
        data = pickle.load(f)
    neurons_list = sorted(list(data.nodes))
    num_neurons = len(neurons_list)
    distance_average_adj_mat = np.zeros((num_neurons, num_neurons))
    num_synapses_in_data = len(data.edges)
    for pre_idx in range(num_neurons):
        for post_idx in range(num_neurons):
            pre = neurons_list[pre_idx]
            post = neurons_list[post_idx]
            distance = np.sqrt(np.sum((data.nodes[pre]['coords'] - data.nodes[post]['coords']) ** 2))
            distance_average_adj_mat[pre_idx, post_idx] = -beta * distance

    types_distances_average_adj_mat = np.exp(distance_average_adj_mat) * er_block_average
    normalization = num_synapses_in_data / types_distances_average_adj_mat.sum()
    types_distances_average_adj_mat *= normalization
    np.clip(types_distances_average_adj_mat, 10e-10, 1-10e-10, out=types_distances_average_adj_mat)
    return types_distances_average_adj_mat
