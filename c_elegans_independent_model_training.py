import pickle
import numpy as np
import networkx as nx
from CElegansNeuronsAdder import CElegansNeuronsAdder
from c_elegans_data_parsing import SUB_COARSE_TYPES_MAPPING
from c_elegans_constants import C_ELEGANS_LENGTHS


def model_log_likelihood(spls, smi, beta, reference_age, developmental_ages, c_elegans_connectome_path):
    with open(c_elegans_connectome_path, 'rb') as f:
        data = pickle.load(f)

    log_likelihood = 0
    mirror_spls_dict = {}
    for synapse in data.edges():
        synapse_data = data.get_edge_data(synapse[0], synapse[1])
        # synapse is a tuple of the form (source, target, attributes) where attributes is a dict.
        synapse_birth_time = synapse_data['birth time']
        if synapse_birth_time >= reference_age:
            continue
        else:
            synapse_age = reference_age - synapse_birth_time
            synapse_length = synapse_data['length']
            pre_type = data.nodes[synapse[0]]['type']
            post_type = data.nodes[synapse[1]]['type']
            synapse_type = (pre_type, post_type)
            if synapse_type not in mirror_spls_dict.keys():
                cur_spls_across_dev = {}
                for dev_stage in spls.keys():
                    cur_spls_across_dev[dev_stage] = spls[dev_stage][synapse_type]
                mirror_spls_dict[synapse_type] = cur_spls_across_dev
            p = p_synapse_is_formed(mirror_spls_dict[synapse_type], smi, beta, developmental_ages, synapse_age,
                                    synapse_birth_time, synapse_length)
            if p == 0:
                log_likelihood += np.log(np.finfo(float).eps)
            else:
                log_likelihood += np.log(p)

    for synapse in nx.non_edges(data):
        # same procedure as above
        synapse_birth_time = max(data.nodes[synapse[0]]['birth_time'], data.nodes[synapse[1]]['birth_time'])
        if synapse_birth_time >= reference_age:
            continue
        else:
            synapse_age = reference_age - synapse_birth_time
            synapse_length = np.sqrt(np.sum((data.nodes[synapse[0]]['coords'] - data.nodes[synapse[1]]['coords']) ** 2))
            pre_type = data.nodes[synapse[0]]['type']
            post_type = data.nodes[synapse[1]]['type']
            synapse_type = (pre_type, post_type)
            if synapse_type not in mirror_spls_dict.keys():
                cur_spls_across_dev = {}
                for dev_stage in spls.keys():
                    cur_spls_across_dev[dev_stage] = spls[dev_stage][synapse_type]
                mirror_spls_dict[synapse_type] = cur_spls_across_dev
            p = p_synapse_is_formed(mirror_spls_dict[synapse_type], smi, beta, developmental_ages, synapse_age,
                                    synapse_birth_time, synapse_length)
            if p == 1:
                log_likelihood += np.log(np.finfo(float).eps)
            else:
                log_likelihood += np.log(1 - p)

    return log_likelihood


# spls is a dict of dicts. The outer dict keys is developmental stages and the inner keys are synaptic
# types. There is a value of S+ for each type at each stage. Here we build a dict containing the values for
# a desired type across all stages.
def convert_spls_dict_to_spls_across_dev(spls, pre_synaptic_type, post_synaptic_type):
    spls_across_dev = {}
    for dev_stage in spls.keys():
        spls_across_dev[dev_stage] = spls[dev_stage][(pre_synaptic_type, post_synaptic_type)]
    return spls_across_dev


# Calculate the current fold growth of the worm's length assuming piecewise linear growth between experimental
# measurements.
def calc_elongation_factor(worm_age):
    ages_of_measured_len = sorted(C_ELEGANS_LENGTHS.keys())
    cur_elongation_stage = 1
    for age in ages_of_measured_len[1:]:
        if worm_age <= age:
            break
        cur_elongation_stage += 1
    previous_measured_age = ages_of_measured_len[cur_elongation_stage - 1]
    next_measured_age = ages_of_measured_len[cur_elongation_stage]
    previous_len = C_ELEGANS_LENGTHS[previous_measured_age]
    next_len = C_ELEGANS_LENGTHS[next_measured_age]

    cur_len = previous_len + (worm_age - previous_measured_age) * (next_len - previous_len) / (
            next_measured_age - previous_measured_age)
    cur_elongation_factor = cur_len / C_ELEGANS_LENGTHS[0]
    return cur_elongation_factor


# Calculates the average of the probabilities of synapses to be formed (the density of the network) according to the
# model when run from time 0 (no neurons) to reference_age of the worm.
def calc_mean_p_synapse_is_formed(spls, smi, beta, reference_age, developmental_ages, c_elegans_connectome_path,
                                  synapse_type=None):
    with open(c_elegans_connectome_path, 'rb') as f:
        data = pickle.load(f)

    mean_p_synapse_is_formed = 0
    num_possible_synapses = 0
    formation_probs = {}
    for synapse in data.edges():
        synapse_data = data.get_edge_data(synapse[0], synapse[1])
        # synapse is a tuple of the form (source, target, attributes) where attributes is a dict.
        synapse_birth_time = synapse_data['birth time']
        if synapse_birth_time >= reference_age:
            continue
        pre_type = data.nodes[synapse[0]]['type']
        post_type = data.nodes[synapse[1]]['type']
        if synapse_type is not None and (synapse_type[0] != pre_type or synapse_type[1] != post_type):
            continue

        spls_across_dev = convert_spls_dict_to_spls_across_dev(spls, pre_type, post_type)

        synapse_age = reference_age - synapse_birth_time
        synapse_length = synapse_data['length']
        p = p_synapse_is_formed(spls_across_dev, smi, beta, developmental_ages, synapse_age, synapse_birth_time,
                                synapse_length)
        mean_p_synapse_is_formed += p
        formation_probs[synapse] = p
        num_possible_synapses += 1

    for synapse in nx.non_edges(data):
        # same procedure as above
        synapse_birth_time = max(data.nodes[synapse[0]]['birth_time'], data.nodes[synapse[1]]['birth_time'])
        if synapse_birth_time >= reference_age:
            continue
        pre_type = data.nodes[synapse[0]]['type']
        post_type = data.nodes[synapse[1]]['type']
        if synapse_type is not None and (synapse_type[0] != pre_type or synapse_type[1] != post_type):
            continue

        spls_across_dev = convert_spls_dict_to_spls_across_dev(spls, pre_type, post_type)

        synapse_age = reference_age - synapse_birth_time
        synapse_length = np.sqrt(np.sum((data.nodes[synapse[0]]['coords'] - data.nodes[synapse[1]]['coords']) ** 2))
        p = p_synapse_is_formed(spls_across_dev, smi, beta, developmental_ages, synapse_age, synapse_birth_time,
                                synapse_length)
        mean_p_synapse_is_formed += p
        formation_probs[synapse] = p
        num_possible_synapses += 1

    mean_num_synapses = mean_p_synapse_is_formed
    if num_possible_synapses != 0:
        mean_p_synapse_is_formed /= num_possible_synapses
    return mean_p_synapse_is_formed, mean_num_synapses, formation_probs


def p_synapse_is_formed_across_ages(spls_across_dev, smi, beta, developmental_ages, time_passed_from_given_state,
                                    time_of_given_state, norm_length,
                                    given_past_state=0, time_step=10):
    synapse_formed_given_age_prob = np.zeros(int(time_passed_from_given_state / time_step) + 1)
    if given_past_state:
        synapse_formed_given_age_prob[0] = 1
    cur_developmental_stage = 0
    spl = spls_across_dev[cur_developmental_stage]
    for n in range(1, synapse_formed_given_age_prob.size):
        # The age of the worm n time steps after the given state of the synapse.
        cur_worm_age = time_of_given_state + n * time_step
        while cur_worm_age > developmental_ages[cur_developmental_stage]:
            cur_developmental_stage += 1
            spl = spls_across_dev[cur_developmental_stage]
        # The probability of the synapse to be formed in the last time step, given it wasn't formed then (taking into
        # account the change in distance between neurons during development)
        elongation_factor = calc_elongation_factor(cur_worm_age)
        c = spl * np.exp(
            -beta * norm_length * elongation_factor)
        # The recursion relation:
        # P(formed_now) = P(formed_time_step_ago)*P(not_erased_time_step_ago) +
        #                 + P(not_formed_time_step_ago)*P(formed_time_step_ago)
        synapse_formed_given_age_prob[n] = c + (1 - smi - c) * synapse_formed_given_age_prob[n - 1]
    return synapse_formed_given_age_prob


def p_synapse_is_formed(spls_across_dev, smi, beta, developmental_ages, time_passed_from_given_state,
                        time_of_given_state, norm_length,
                        given_past_state=0, time_step=10):
    synapse_formed_given_age_prob = p_synapse_is_formed_across_ages(spls_across_dev, smi, beta, developmental_ages,
                                                                    time_passed_from_given_state,
                                                                    time_of_given_state, norm_length,
                                                                    given_past_state=given_past_state,
                                                                    time_step=time_step)
    return synapse_formed_given_age_prob[-1]


def calc_model_adj_mat(spls, smi, beta, reference_age, developmental_ages, c_elegans_connectome_path):
    with open(c_elegans_connectome_path, 'rb') as f:
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
            # spls is a dict of dicts. The outer dict keys is developmental stages and the inner keys are synaptic
            # types. There is a value of S+ for each type at each stage. Here we build a dict containing the values for
            # a desired type across all stages.
            spls_across_dev = {}
            for dev_stage in spls.keys():
                spls_across_dev[dev_stage] = spls[dev_stage][(pre_type, post_type)]

            synapse_age = reference_age - synapse_birth_time
            synapse_length = np.sqrt(np.sum((data.nodes[pre]['coords'] - data.nodes[post]['coords']) ** 2))
            p = p_synapse_is_formed(spls_across_dev, smi, beta, developmental_ages, synapse_age, synapse_birth_time,
                                    synapse_length)
            adj_mat[pre_idx, post_idx] = p
            post_idx += 1
        pre_idx += 1

    return adj_mat


def count_synapses_of_type(synaptic_type, type_configuration, data_path=None, connectome=None):
    if connectome is None:
        with open(data_path, 'rb') as f:
            connectome = pickle.load(f)
    if synaptic_type == (None, None):
        num_synapses = len(connectome.edges())
    else:
        num_synapses = 0
        neurons = connectome.nodes()
        for synapse in connectome.edges():
            pre_syn = synapse[0]
            post_syn = synapse[1]
            if CElegansNeuronsAdder.COARSE_TYPES == type_configuration:
                cur_synapse_coarse_type = (
                    SUB_COARSE_TYPES_MAPPING[neurons[pre_syn]["type"]],
                    SUB_COARSE_TYPES_MAPPING[neurons[post_syn]["type"]])
                if cur_synapse_coarse_type == synaptic_type:
                    num_synapses += 1
            elif CElegansNeuronsAdder.SUB_TYPES == type_configuration or \
                    CElegansNeuronsAdder.ARTIFICIAL_TYPES == type_configuration:
                if neurons[pre_syn]["type"] == synaptic_type[0] and neurons[post_syn]["type"] == synaptic_type[1]:
                    num_synapses += 1

    return num_synapses


def count_neurons_of_type(neuron_type, reference_age, neurons_subset_path, type_configuration,
                          c_elegans_data_path=CElegansNeuronsAdder.DEFAULT_DATA_PATH,
                          types_file_name=CElegansNeuronsAdder.DEFAULT_TYPES_FILE_NAME):
    # use the CElegansNeuronsAdder class to create a df of neurons with their data
    dummy = CElegansNeuronsAdder(type_configuration, neurons_sub_set_path=neurons_subset_path,
                                 c_elegans_data_path=c_elegans_data_path, types_file_name=types_file_name)
    neurons_df = dummy.get_neurons_df()
    # take only the neurons that are born before the reference age.
    neurons_df = neurons_df.loc[neurons_df[CElegansNeuronsAdder.BIRTH_TIME_STR] <= reference_age]
    neurons_of_desired_type = [neuron for neuron in neurons_df.index if
                               neurons_df.loc[neuron, CElegansNeuronsAdder.TYPE_STR] == neuron_type]
    num_neurons = len(neurons_of_desired_type)
    return num_neurons


def calc_synaptic_type_factor(synaptic_type, beta, smi, reference_age, developmental_ages, spls_across_dev,
                              type_configuration, data_path, tolerance=0.05):
    new_dev_stage = 0
    while reference_age > developmental_ages[new_dev_stage]:
        new_dev_stage += 1
    cur_interval = np.array([0.0, 1.0])
    spls_across_dev[new_dev_stage][synaptic_type] = cur_interval[1] / 2
    num_synapses_in_data = count_synapses_of_type(synaptic_type, type_configuration, data_path=data_path)
    if num_synapses_in_data == 0:
        return 0, 0
    _, expected_num_synapses_model, _ = calc_mean_p_synapse_is_formed(spls_across_dev, smi, beta, reference_age,
                                                                      developmental_ages, data_path,
                                                                      synapse_type=synaptic_type)
    # Perform binary search for the value of S+ that gives on average the right amount of synapses as in the data.
    iteration = 0
    while np.abs(num_synapses_in_data - expected_num_synapses_model) > tolerance:
        if num_synapses_in_data > expected_num_synapses_model:
            cur_interval[0] = spls_across_dev[new_dev_stage][synaptic_type]
            spls_across_dev[new_dev_stage][synaptic_type] = (cur_interval[0] + cur_interval[1]) / 2
        else:
            cur_interval[1] = spls_across_dev[new_dev_stage][synaptic_type]
            spls_across_dev[new_dev_stage][synaptic_type] = (cur_interval[0] + cur_interval[1]) / 2
        _, expected_num_synapses_model, _ = calc_mean_p_synapse_is_formed(spls_across_dev, smi, beta, reference_age,
                                                                          developmental_ages, data_path,
                                                                          synapse_type=synaptic_type)
        iteration += 1
        if spls_across_dev[new_dev_stage][synaptic_type] <= 10e-10 or \
                spls_across_dev[new_dev_stage][synaptic_type] >= 1 - 10e-10:
            # Probably the number of synapses can't be reached with the given S- and beta which are not trainable.
            break
    return spls_across_dev[new_dev_stage][synaptic_type], expected_num_synapses_model


def average_matrix_log_likelihood(av_mat, connectome_path):
    with open(connectome_path, 'rb') as f:
        connectome = pickle.load(f)
    sorted_neurons = sorted(list(connectome.nodes))
    connectome_adj_mat = nx.to_numpy_array(connectome, nodelist=sorted_neurons)
    num_neurons = len(sorted_neurons)
    log_likelihood = 0
    eps = np.finfo('float').eps
    for i in range(num_neurons):
        for j in range(num_neurons):
            if 1 == connectome_adj_mat[i, j]:
                if av_mat[i, j] > 0:
                    log_likelihood += np.log(av_mat[i, j])
                else:
                    log_likelihood += np.log(eps)
            else:
                if av_mat[i, j] < 1:
                    log_likelihood += np.log(1 - av_mat[i, j])
                else:
                    log_likelihood += np.log(eps)
    return log_likelihood


def sample_from_average_adj_mat(av_adj_mat):
    sampled_mat = np.zeros(av_adj_mat.shape)
    for row in range(av_adj_mat.shape[0]):
        for column in range(av_adj_mat.shape[1]):
            r = np.random.rand()
            if r <= av_adj_mat[row, column]:
                sampled_mat[row, column] = 1
    return sampled_mat


def convert_spls_dict_to_mat(spls_path, developmental_stage):
    with open(spls_path, 'rb') as f:
        spls = pickle.load(f)
    spls_for_stage = spls[developmental_stage]
    num_types = int(np.sqrt(len(spls_for_stage.keys())))
    spls_mat = np.zeros((num_types, num_types))
    neuronal_types = []
    for syn_type in spls_for_stage.keys():
        if syn_type[0] not in neuronal_types:
            neuronal_types.append(syn_type[0])
    neuronal_types.sort()
    for syn_type in spls_for_stage.keys():
        spls_mat[neuronal_types.index(syn_type[0]), neuronal_types.index(syn_type[1])] = spls_for_stage[syn_type]
    return spls_mat, neuronal_types


# Works only for inferred types
def convert_spls_mat_to_dict(spls_mat):
    spls_dict = {0: {}}
    for row in range(spls_mat.shape[0]):
        for col in range(spls_mat.shape[1]):
            spls_dict[0][(row, col)] = spls_mat[row, col]
    return spls_dict
