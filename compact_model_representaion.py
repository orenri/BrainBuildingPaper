import numpy as np
import os
import pickle
import networkx as nx
from sklearn.metrics import roc_auc_score

from c_elegans_independent_model_training import convert_spls_dict_to_mat, \
    convert_spls_mat_to_dict, calc_model_adj_mat
from c_elegans_constants import ADULT_WORM_AGE, SINGLE_DEVELOPMENTAL_AGE


def calc_variance(numbers):
    return ((numbers - numbers.mean()) ** 2).mean()


def find_minimal_weighted_variance_sum_partition_sorted(numbers, num_classes):
    sorted_numbers = np.sort(numbers)
    optimal_partitions = {}
    n = len(sorted_numbers)
    for k in range(1, num_classes + 1):
        for i in range(k, n + 1):
            if k == 1:
                optimal_partitions[(k, i)] = (i * calc_variance(sorted_numbers[:i]), [i])
            elif k == i:
                optimal_partitions[(k, i)] = (0, [1] * i)
            else:
                min_variance = np.inf
                min_partition = None
                for j in range(k - 1, i):
                    cur_optional_variance = optimal_partitions[(k - 1, j)][0] + (i - j) * calc_variance(
                        sorted_numbers[j:i])
                    if cur_optional_variance < min_variance:
                        min_variance = cur_optional_variance
                        min_partition = optimal_partitions[(k - 1, j)][1] + [i - j]
                optimal_partitions[(k, i)] = (min_variance, min_partition)
    sizes_of_optimal_subgroups = optimal_partitions[(num_classes, n)][1]
    cur_idx = 0
    optimal_sub_arrays = []
    for size in sizes_of_optimal_subgroups:
        optimal_sub_arrays.append(sorted_numbers[cur_idx:cur_idx + size])
        cur_idx += size

    return optimal_partitions[(num_classes, n)][0], optimal_sub_arrays


def construct_compact_spls_from_classes(full_spls_mat, split_to_classes):
    num_types = full_spls_mat.shape[0]
    compact_spls_representation_mat = np.zeros(full_spls_mat.shape)
    for row in range(num_types):
        for col in range(num_types):
            cur_spl_value = full_spls_mat[row, col]
            if cur_spl_value == 0:
                cur_compact_value = 0
            else:
                for sub_array in split_to_classes:
                    if cur_spl_value in sub_array:
                        cur_compact_value = sub_array.mean()
            compact_spls_representation_mat[row, col] = cur_compact_value
    return compact_spls_representation_mat


def find_spls_compact_representation(spls_path, developmental_stage, unexplained_variance_ratio_thr):
    spls_mat, _ = convert_spls_dict_to_mat(spls_path, developmental_stage)
    non_zeros_spls = spls_mat[np.nonzero(spls_mat)]
    spls_variance = calc_variance(spls_mat.flatten())
    cur_unexplained_variance_ratio = np.inf
    num_classes = 0
    while cur_unexplained_variance_ratio > unexplained_variance_ratio_thr:
        num_classes += 1
        cur_unexplained_variance, cur_split = find_minimal_weighted_variance_sum_partition_sorted(non_zeros_spls,
                                                                                                  num_classes)
        if spls_variance == 0 and cur_unexplained_variance == 0:
            break
        cur_unexplained_variance_ratio = cur_unexplained_variance / spls_variance
    # for the class of zeros
    if 0 in spls_mat:
        num_classes += 1
    compact_spls_representation_mat = construct_compact_spls_from_classes(spls_mat, cur_split)
    return num_classes, convert_spls_mat_to_dict(compact_spls_representation_mat)


def find_compression_to_k_values(spls_path, num_classes, developmental_stage=0):
    spls_mat, _ = convert_spls_dict_to_mat(spls_path, developmental_stage)
    non_zeros_spls = spls_mat[np.nonzero(spls_mat)]
    if non_zeros_spls.size != spls_mat.size:
        num_classes_non_zero = num_classes - 1
    else:
        num_classes_non_zero = num_classes
    _, split = find_minimal_weighted_variance_sum_partition_sorted(non_zeros_spls, num_classes_non_zero)
    compact_spls_mat = construct_compact_spls_from_classes(spls_mat, split)
    return convert_spls_mat_to_dict(compact_spls_mat)


def find_compact_representation_within_performance_tolerance(smi, beta, spls_dir_path, train_data_path,
                                                             test_data_path, tolerance_fraction):
    spls_path = os.path.join(spls_dir_path, f"spls_smi{smi:.5f}_beta{beta:.5f}.pkl")
    with open(spls_path, 'rb') as f:
        full_spls = pickle.load(f)
    full_model_average_mat = calc_model_adj_mat(full_spls, smi, beta, ADULT_WORM_AGE, SINGLE_DEVELOPMENTAL_AGE,
                                                train_data_path)
    with open(test_data_path, 'rb') as f:
        test_data = pickle.load(f)
    test_data_mat = nx.to_numpy_array(test_data, nodelist=sorted(test_data.nodes))
    test_data_mat = test_data_mat.astype(int)
    full_model_performance = roc_auc_score(test_data_mat.flatten(), full_model_average_mat.flatten())
    performance_threshold = full_model_performance * (1 - tolerance_fraction)

    cur_interval = np.array([0, 0.1])
    cur_unex_var_thr = cur_interval.mean()
    cur_compact_size, cur_compact_representation = find_spls_compact_representation(spls_path, 0, cur_unex_var_thr)
    cur_compact_model_av_mat = calc_model_adj_mat(cur_compact_representation, smi, beta, ADULT_WORM_AGE,
                                                  SINGLE_DEVELOPMENTAL_AGE,
                                                  train_data_path)
    cur_compact_performance = roc_auc_score(test_data_mat.flatten(), cur_compact_model_av_mat.flatten())
    while cur_compact_performance < performance_threshold or cur_interval[1] - cur_interval[0] > 10e-10:
        if cur_compact_performance < performance_threshold:
            cur_interval[1] = cur_unex_var_thr
            cur_unex_var_thr = cur_interval.mean()
        else:
            cur_interval[0] = cur_unex_var_thr
            cur_unex_var_thr = cur_interval.mean()

        cur_compact_size, cur_compact_representation = find_spls_compact_representation(spls_path, 0, cur_unex_var_thr)
        cur_compact_model_av_mat = calc_model_adj_mat(cur_compact_representation, smi, beta, ADULT_WORM_AGE,
                                                      SINGLE_DEVELOPMENTAL_AGE,
                                                      train_data_path)
        cur_compact_performance = roc_auc_score(test_data_mat.flatten(), cur_compact_model_av_mat.flatten())

    return cur_compact_size, cur_compact_representation, cur_compact_model_av_mat, cur_unex_var_thr
