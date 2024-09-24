import numpy as np
import pickle
from copy import deepcopy
from itertools import product
import os

from c_elegans_independent_model_training import calc_elongation_factor, convert_spls_dict_to_spls_across_dev, \
    count_synapses_of_type, average_matrix_log_likelihood
from c_elegans_reciprocal_model_training import single_binary_search_update
from c_elegans_constants import ADULT_WORM_AGE, SINGLE_DEVELOPMENTAL_AGE
from CElegansNeuronsAdder import CElegansNeuronsAdder


def p_synapse_is_formed_syn_count_model(spls_across_dev, smi, beta, developmental_ages, time_passed_from_given_state,
                                        time_of_given_state, norm_length, given_past_state=0, time_step=10):
    # Initialize the matrix of pmf of the number of synapses across time (the i,j-th entry of
    # `num_synapses_pmf_across_time` is the probability to see j synapses at timestep i).
    num_time_steps_for_calculation = int(time_passed_from_given_state / time_step)
    max_num_synapses_at_the_end = given_past_state + num_time_steps_for_calculation
    num_synapses_pmf_across_time = np.zeros((num_time_steps_for_calculation + 1, max_num_synapses_at_the_end + 1))
    num_synapses_pmf_across_time[0, given_past_state] = 1

    cur_developmental_stage = 0
    spl = spls_across_dev[cur_developmental_stage]
    for step in range(1, num_time_steps_for_calculation + 1):
        # The age of the worm n time steps after the given state of the synapse.
        cur_worm_age = time_of_given_state + step * time_step
        while cur_worm_age > developmental_ages[cur_developmental_stage]:
            cur_developmental_stage += 1
            spl = spls_across_dev[cur_developmental_stage]

        elongation_factor = calc_elongation_factor(cur_worm_age)
        formation_prob = spl * np.exp(-beta * norm_length * elongation_factor)
        for syn_count in range(max_num_synapses_at_the_end + 1):
            # The number of synapses can be changed only by 1 at each step, so the probability to be further from a
            # given state than the number of steps passed from its corresponding time is 0.
            if abs(syn_count - given_past_state) > step:
                continue

            # Don't form a connection.
            prob_of_staying_the_same = num_synapses_pmf_across_time[step - 1, syn_count] * (
                    1 - formation_prob)
            # If there are connections
            if syn_count > 0:
                # Don't prune exising ones (this refers to the case that a connection was not formed, initialized
                # above).
                prob_of_staying_the_same *= (1 - smi)
                # Form and prune ones
                prob_of_staying_the_same += num_synapses_pmf_across_time[step - 1, syn_count] * formation_prob * smi

            # Form a connection
            if syn_count > 0:
                prob_of_adding_connection = num_synapses_pmf_across_time[step - 1, syn_count - 1] * formation_prob
                # If there were connections before, don't prune them
                if syn_count - 1 > 0:
                    prob_of_adding_connection *= (1 - smi)
            # It is not possible to reach 0 synapses by adding a connection
            else:
                prob_of_adding_connection = 0

            if syn_count < max_num_synapses_at_the_end:
                # Remove the connection and don't form another one
                prob_of_removing_connection = num_synapses_pmf_across_time[step - 1, syn_count + 1] * smi * (
                        1 - formation_prob)
            # It is not possible to reach the maximal number of synapses by removing a connection (this is only
            # reachable if a connection was added in every interation, and never pruned).
            else:
                prob_of_removing_connection = 0

            num_synapses_pmf_across_time[
                step, syn_count] = prob_of_staying_the_same + prob_of_adding_connection + prob_of_removing_connection

    # assert (np.abs(num_synapses_pmf_across_time.sum(axis=1) - 1).max() < 10 ** -14)
    # The probability to have at least one synapse at the final step
    # return num_synapses_pmf_across_time[-1, 1:].sum()
    return 1 - num_synapses_pmf_across_time[-1, 0]


def calc_expected_num_synapses_type_pair(neuronal_types_pair, smi, beta, reference_age, developmental_ages, spls,
                                         data_path):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    sorted_neurons = sorted(data.nodes)
    num_neurons = len(sorted_neurons)
    expected_num_synapses = 0
    spls_across_dev = convert_spls_dict_to_spls_across_dev(spls, neuronal_types_pair[0], neuronal_types_pair[1])
    for pre_idx in range(num_neurons):
        pre_neuron = sorted_neurons[pre_idx]
        pre_type = data.nodes[pre_neuron]['type']
        if neuronal_types_pair[0] != pre_type:
            continue
        for post_idx in range(num_neurons):
            if pre_idx == post_idx:
                continue
            post_neuron = sorted_neurons[post_idx]
            post_type = data.nodes[post_neuron]['type']
            if neuronal_types_pair[1] != post_type:
                continue
            synapse_birth_time = max(data.nodes[pre_neuron]['birth_time'], data.nodes[post_neuron]['birth_time'])
            if synapse_birth_time >= reference_age:
                continue
            synapse_age = reference_age - synapse_birth_time
            norm_dist_between_neurons = np.linalg.norm(
                data.nodes[pre_neuron]['coords'] - data.nodes[post_neuron]['coords'])

            expected_num_synapses += p_synapse_is_formed_syn_count_model(spls_across_dev, smi, beta, developmental_ages,
                                                                         synapse_age, synapse_birth_time,
                                                                         norm_dist_between_neurons)

    return expected_num_synapses


def search_spl_syn_count_model(neuronal_types_pair, smi, beta, reference_age, developmental_ages, spls,
                               type_configuration, data_path, tolerance=0.05):
    new_dev_stage = 0
    while reference_age > developmental_ages[new_dev_stage]:
        new_dev_stage += 1

    spls_copy = deepcopy(spls)
    cur_search_window = np.array([0, 1])
    spls_copy[new_dev_stage][neuronal_types_pair] = (cur_search_window[1] + cur_search_window[0]) / 2
    num_synapses_in_data = count_synapses_of_type(neuronal_types_pair, type_configuration, data_path=data_path)
    expected_num_synapses_model = calc_expected_num_synapses_type_pair(neuronal_types_pair, smi, beta, reference_age,
                                                                       developmental_ages, spls_copy, data_path)
    iteration = 0
    while abs(expected_num_synapses_model - num_synapses_in_data) > tolerance:
        spls_copy[new_dev_stage][neuronal_types_pair] = single_binary_search_update(num_synapses_in_data,
                                                                                    expected_num_synapses_model,
                                                                                    cur_search_window)
        expected_num_synapses_model = calc_expected_num_synapses_type_pair(neuronal_types_pair, smi, beta,
                                                                           reference_age,
                                                                           developmental_ages, spls_copy, data_path)
        if iteration > 2 * 10 * np.log(10) / np.log(2):
            # Probably the number of synapses can't be reached up to the tolerance with the given parameters, as the
            # size of the square side is already smaller than 10e-10.
            break
        iteration += 1
    return spls_copy[new_dev_stage][neuronal_types_pair]


def calc_syn_count_model_average_mat(smi, beta, spls, reference_age, developmental_ages, data_path):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    sorted_nodes = sorted(list(data.nodes))
    sorted_nodes_before_reference_age = []
    for neuron in sorted_nodes:
        if data.nodes[neuron]['birth_time'] < reference_age:
            sorted_nodes_before_reference_age.append(neuron)
    num_neurons = len(sorted_nodes_before_reference_age)
    adj_mat = np.zeros((num_neurons, num_neurons))
    pre_idx = 0
    for pre in sorted_nodes_before_reference_age:
        post_idx = 0
        for post in sorted_nodes_before_reference_age:
            if pre == post:
                post_idx += 1
                continue
            synapse_birth_time = max(data.nodes[pre]['birth_time'], data.nodes[post]['birth_time'])
            pre_type = data.nodes[pre]['type']
            post_type = data.nodes[post]['type']
            spls_across_dev = convert_spls_dict_to_spls_across_dev(spls, pre_type, post_type)

            synapse_age = reference_age - synapse_birth_time
            synapse_length = np.linalg.norm(data.nodes[pre]['coords'] - data.nodes[post]['coords'])
            p = p_synapse_is_formed_syn_count_model(spls_across_dev, smi, beta, developmental_ages, synapse_age,
                                                    synapse_birth_time, synapse_length)
            adj_mat[pre_idx, post_idx] = p
            post_idx += 1
        pre_idx += 1

    return adj_mat


def train_single_epoch_syn_count_model_distributed():
    func_id = int(os.environ['LSB_JOBINDEX']) - 1
    num_types = func_id // 1600 + 1
    func_id = func_id % 1600
    cur_path = os.getcwd()
    params_path = os.path.join(cur_path, "ParamFiles", "spl_per_type_params_low_smi_low_beta.pkl")
    with open(params_path, 'rb') as f:
        params = pickle.load(f)
    smi = params[func_id, 0]
    beta = params[func_id, 1]
    train_data_path = os.path.join(cur_path, "CElegansData", "InferredTypes", "connectomes", f"{num_types}_types",
                                   "Dataset7.pkl")
    out_spl_path = os.path.join(cur_path, "SavedOutputs", "SynCountModel", "S+s", f"{num_types}_types")
    os.makedirs(out_spl_path, exist_ok=True)
    out_like_path = os.path.join(cur_path, "SavedOutputs", "SynCountModel", "likelihoods", f"{num_types}_types")
    os.makedirs(out_like_path, exist_ok=True)
    out_av_mat_path = os.path.join(cur_path, "SavedOutputs", "SynCountModel", "average_adj_mats", f"{num_types}_types")
    os.makedirs(out_av_mat_path, exist_ok=True)
    spls = {0: {i: 0 for i in list(product(list(range(num_types)), list(range(num_types))))}}
    for type_pair in spls[0].keys():
        spls[0][type_pair] = search_spl_syn_count_model(type_pair, smi, beta, ADULT_WORM_AGE, SINGLE_DEVELOPMENTAL_AGE,
                                                        spls, CElegansNeuronsAdder.ARTIFICIAL_TYPES, train_data_path)
    with open(out_spl_path, 'wb') as f:
        pickle.dump(spls, f)

    av_mat = calc_syn_count_model_average_mat(smi, beta, spls, ADULT_WORM_AGE, SINGLE_DEVELOPMENTAL_AGE, train_data_path)
    with open(out_av_mat_path, 'wb') as f:
        pickle.dump(av_mat, f)

    log_like = average_matrix_log_likelihood(av_mat, train_data_path)
    log_like_file_path = os.path.join(out_like_path, f"{func_id}.csv")
    with open(log_like_file_path, 'w') as f:
        f.write(f"smi,beta,log-likelihood\n{smi},{beta},{log_like}\n")


def main():
    train_single_epoch_syn_count_model_distributed()
    # num_types = 8
    # data_path = f"CElegansData\\InferredTypes\\connectomes\\{num_types}_types\\Dataset7.pkl"
    # smi = 0.0375
    # beta = 0.0125
    # spls = {0: {i: 0 for i in list(product(list(range(num_types)), list(range(num_types))))}}
    # for type_pair in spls[0].keys():
    #     spls[0][type_pair] = search_spl_syn_count_model(type_pair, smi, beta, ADULT_WORM_AGE, SINGLE_DEVELOPMENTAL_AGE,
    #                                                     spls, CElegansNeuronsAdder.ARTIFICIAL_TYPES, data_path)
    # with open("D:\OrenRichter\\temp\syn_count_model\\spls.pkl", 'wb') as f:
    #     pickle.dump(spls, f)
    #
    # av_mat = calc_syn_count_model_average_mat(smi, beta, spls, ADULT_WORM_AGE, SINGLE_DEVELOPMENTAL_AGE, data_path)
    # with open("D:\OrenRichter\\temp\syn_count_model\\av_mat.pkl", 'wb') as f:
    #     pickle.dump(av_mat, f)


if __name__ == "__main__":
    main()
