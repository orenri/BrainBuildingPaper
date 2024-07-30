import pickle
import numpy as np
import networkx as nx
import os
from c_elegans_independent_model_training import calc_elongation_factor, count_synapses_of_type, \
    convert_spls_dict_to_spls_across_dev
from c_elegans_constants import MEI_ZHEN_NUM_NEURONS
from wrap_cluster_runs import find_max_likelihood_full_model

RECIPROCAL_DYAD_IDX = 0
ONLY_UPPER_TRIANGLE_SYNAPSE_IDX = 1
ONLY_LOWER_TRIANGLE_SYNAPSE_IDX = 2
EMPTY_DYAD_IDX = 3


def calc_joint_reciprocal_synapse_distribution(spls_across_dev_1, spls_across_dev_2, smi, beta, gamma,
                                               developmental_ages, time_passed_from_given_state, time_of_given_state,
                                               norm_length, given_past_state=np.array([0, 0]), time_step=10):
    num_time_steps = int(time_passed_from_given_state / time_step)
    only_synapse_1_formed_given_age_prob = np.zeros(num_time_steps + 1)
    only_synapse_2_formed_given_age_prob = np.zeros(num_time_steps + 1)
    reciprocal_dyad_formed_given_age_prob = np.zeros(num_time_steps + 1)
    no_synapse_formed_given_age_prob = np.zeros(num_time_steps + 1)
    if given_past_state[0] and not given_past_state[1]:
        only_synapse_1_formed_given_age_prob[0] = 1
    elif not given_past_state[0] and given_past_state[1]:
        only_synapse_2_formed_given_age_prob[0] = 1
    elif given_past_state[0] and given_past_state[1]:
        reciprocal_dyad_formed_given_age_prob[0] = 1
    else:
        no_synapse_formed_given_age_prob[0] = 1
    cur_developmental_stage = 0
    spl1 = spls_across_dev_1[cur_developmental_stage]
    spl2 = spls_across_dev_2[cur_developmental_stage]
    for n in range(1, num_time_steps + 1):
        # The age of the worm n time steps after the given state of the synapse.
        cur_worm_age = time_of_given_state + n * time_step
        while cur_worm_age > developmental_ages[cur_developmental_stage]:
            cur_developmental_stage += 1
            spl1 = spls_across_dev_1[cur_developmental_stage]
            spl2 = spls_across_dev_2[cur_developmental_stage]
        # The probability of the synapses 1, 2 to be formed now, given they weren't formed before
        # (taking into account the length of the synapse assuming proper elongation during development)
        elongation_factor = calc_elongation_factor(cur_worm_age)
        c1 = spl1 * np.exp(-beta * norm_length * elongation_factor)
        c2 = spl2 * np.exp(-beta * norm_length * elongation_factor)
        # The recursion relation:
        # (the formation of a reciprocal synapse to an already existing one is multiplied by gamma)
        #
        # P(only_1_formed) = P(reciprocal_dyad_formed_time_step_ago) * P(1_not_erased_now) * P(2_erased_now) +
        # + P(1_formed_time_step_ago|2_not_formed_time_step_ago) * P(1_not_erased_now) * (1 - gamma * P(2_formed_now)) +
        # + P(no_synapse_formed_time_step_ago) * P(1_formed_now) * P(2_not_formed_now) +
        # + P(1_not_formed_time_step_ago|2_formed_time_step_ago) * gamma * P(1_formed_now) * P(2_erased_now)
        #
        # P(reciprocal_dyad_formed) = P(no_synapse_formed_time_step_ago) * P(1_formed_now) * P(2_formed_now) +
        #       + P(reciprocal_dyad_formed_time_step_ago) * P(1_not_erased_now) * P(2_not_erased_now) +
        #       + P(1_formed_time_step_ago|2_not_formed_time_step_ago) * gamma * P(2_formed_now) * P(1_not_erased_now) +
        #       + P(2_formed_time_step_ago|1_not_formed_time_step_ago) * gamma * P(1_formed_now) * P(2_not_erased_now)

        reciprocal_dyad_formed_given_age_prob[n] = reciprocal_dyad_formed_given_age_prob[n - 1] * (1 - smi) ** 2 + \
                                                   only_synapse_1_formed_given_age_prob[n - 1] * (
                                                           1 - smi) * min(gamma * c2, 1) + \
                                                   only_synapse_2_formed_given_age_prob[n - 1] * min(gamma * c1, 1) * (
                                                           1 - smi) + no_synapse_formed_given_age_prob[
                                                       n - 1] * c1 * c2

        only_synapse_1_formed_given_age_prob[n] = reciprocal_dyad_formed_given_age_prob[n - 1] * (1 - smi) * smi + \
                                                  only_synapse_1_formed_given_age_prob[n - 1] * (1 - smi) * (
                                                          1 - min(gamma * c2, 1)) + \
                                                  only_synapse_2_formed_given_age_prob[
                                                      n - 1] * min(gamma * c1, 1) * smi + \
                                                  no_synapse_formed_given_age_prob[
                                                      n - 1] * c1 * (1 - c2)

        only_synapse_2_formed_given_age_prob[n] = reciprocal_dyad_formed_given_age_prob[n - 1] * smi * (1 - smi) + \
                                                  only_synapse_1_formed_given_age_prob[n - 1] * smi * min(gamma * c2,
                                                                                                          1) + \
                                                  only_synapse_2_formed_given_age_prob[n - 1] * (
                                                          1 - min(gamma * c1, 1)) * (
                                                          1 - smi) + no_synapse_formed_given_age_prob[n - 1] * (
                                                          1 - c1) * c2

        no_synapse_formed_given_age_prob[n] = reciprocal_dyad_formed_given_age_prob[n - 1] * smi ** 2 + \
                                              only_synapse_1_formed_given_age_prob[n - 1] * smi * (
                                                      1 - min(gamma * c2, 1)) + \
                                              only_synapse_2_formed_given_age_prob[n - 1] * (
                                                      1 - min(gamma * c1, 1)) * smi + \
                                              no_synapse_formed_given_age_prob[n - 1] * (1 - c1) * (1 - c2)
    return reciprocal_dyad_formed_given_age_prob[-1], only_synapse_1_formed_given_age_prob[-1], \
        only_synapse_2_formed_given_age_prob[-1], no_synapse_formed_given_age_prob[-1]


def calc_reciprocal_dependence_model_dyads_states_distribution(spls, smi, beta, gamma, reference_age,
                                                               developmental_ages,
                                                               c_elegans_connectome_path):
    with open(c_elegans_connectome_path, 'rb') as f:
        data = pickle.load(f)

    sorted_nodes = sorted(list(data.nodes))
    sorted_nodes_before_reference_age = []
    for neuron in sorted_nodes:
        if data.nodes[neuron]['birth_time'] < reference_age:
            sorted_nodes_before_reference_age.append(neuron)
    num_neurons = len(sorted_nodes_before_reference_age)
    dyads_distributions = {}
    for pre_idx in range(num_neurons - 1):
        pre = sorted_nodes_before_reference_age[pre_idx]
        for post_idx in range(pre_idx + 1, num_neurons):
            post = sorted_nodes_before_reference_age[post_idx]
            synapse_birth_time = max(data.nodes[pre]['birth_time'], data.nodes[post]['birth_time'])
            pre_type = data.nodes[pre]['type']
            post_type = data.nodes[post]['type']
            # spls is a dict of dicts. The outer dict keys is developmental stages and the inner keys are synaptic
            # types. There is a value of S+ for each type at each stage. Here we build a dict containing the values for
            # a desired type across all stages.
            spls_across_dev_pre_post = {}
            spls_across_dev_post_pre = {}
            for dev_stage in spls.keys():
                spls_across_dev_pre_post[dev_stage] = spls[dev_stage][(pre_type, post_type)]
                spls_across_dev_post_pre[dev_stage] = spls[dev_stage][(post_type, pre_type)]

            synapse_age = reference_age - synapse_birth_time
            synapse_length = np.sqrt(np.sum((data.nodes[pre]['coords'] - data.nodes[post]['coords']) ** 2))
            p_reciprocal, p_only_i_j, p_only_j_i, p_no_synapse = calc_joint_reciprocal_synapse_distribution(
                spls_across_dev_pre_post, spls_across_dev_post_pre, smi, beta, gamma,
                developmental_ages, synapse_age, synapse_birth_time, synapse_length)
            dyads_distributions[(pre_idx, post_idx)] = (p_reciprocal, p_only_i_j, p_only_j_i, p_no_synapse)
    return dyads_distributions


def calc_reciprocal_dependence_model_dyads_states_distribution_syn_list(spls, smi, beta, gamma, reference_age,
                                                                        developmental_ages, syn_list_path):
    with open(syn_list_path, 'rb') as f:
        syn_list = pickle.load(f)

    dyads_distributions = {}
    for synapse in syn_list:
        synapse_birth_time = synapse[-1]['birth time']
        if synapse_birth_time >= reference_age:
            continue
        forward_key = tuple(sorted([synapse[0], synapse[1]]))
        if (forward_key[1], forward_key[0]) in dyads_distributions.keys():
            continue
        if forward_key == (synapse[0], synapse[1]):
            forward_type = synapse[-1]['type']
        else:
            forward_type = (synapse[-1]['type'][1], synapse[-1]['type'][0])
        # spls is a dict of dicts. The outer dict keys is developmental stages and the inner keys are synaptic
        # types. There is a value of S+ for each type at each stage. Here we build a dict containing the values for
        # a desired type across all stages.
        spls_across_dev_forward = {}
        spls_across_dev_reciprocal = {}
        for dev_stage in spls.keys():
            spls_across_dev_forward[dev_stage] = spls[dev_stage][forward_type]
            spls_across_dev_reciprocal[dev_stage] = spls[dev_stage][(forward_type[1], forward_type[0])]

        synapse_age = reference_age - synapse_birth_time
        synapse_length = synapse[-1]['length']
        p_reciprocal, p_only_i_j, p_only_j_i, p_no_synapse = calc_joint_reciprocal_synapse_distribution(
            spls_across_dev_forward, spls_across_dev_reciprocal, smi, beta, gamma,
            developmental_ages, synapse_age, synapse_birth_time, synapse_length)
        dyads_distributions[forward_key] = (p_reciprocal, p_only_i_j, p_only_j_i, p_no_synapse)
    return dyads_distributions


def calc_reciprocal_dependence_model_log_likelihood(spls, smi, beta, gamma, reference_age, developmental_ages,
                                                    c_elegans_connectome_path):
    with open(c_elegans_connectome_path, 'rb') as f:
        data = pickle.load(f)
    data_adj_mat = nx.to_numpy_array(data, nodelist=sorted(data.nodes))
    data_adj_mat = data_adj_mat.astype(int)
    num_neurons = data_adj_mat.shape[0]
    dyads_distributions = calc_reciprocal_dependence_model_dyads_states_distribution(spls, smi, beta, gamma,
                                                                                     reference_age,
                                                                                     developmental_ages,
                                                                                     c_elegans_connectome_path)
    log_likelihood = 0
    for i in range(num_neurons - 1):
        for j in range(i + 1, num_neurons):
            if data_adj_mat[i, j] and data_adj_mat[j, i]:
                log_likelihood += np.log(dyads_distributions[(i, j)][RECIPROCAL_DYAD_IDX])
            elif data_adj_mat[i, j] and not data_adj_mat[j, i]:
                log_likelihood += np.log(dyads_distributions[(i, j)][ONLY_UPPER_TRIANGLE_SYNAPSE_IDX])
            elif not data_adj_mat[i, j] and data_adj_mat[j, i]:
                log_likelihood += np.log(dyads_distributions[(i, j)][ONLY_LOWER_TRIANGLE_SYNAPSE_IDX])
            else:
                log_likelihood += np.log(dyads_distributions[(i, j)][EMPTY_DYAD_IDX])
    return log_likelihood


def calc_reciprocal_dependence_model_log_likelihood_from_dyads_dist_string_keys(dyads_distributions, syn_list_path):
    with open(syn_list_path, 'rb') as f:
        syn_list = pickle.load(f)
    log_likelihood = 0
    iterated_forward_keys = []
    for syn in syn_list:
        forward_key = tuple(sorted([syn[0], syn[1]]))
        if forward_key in iterated_forward_keys:
            continue
        iterated_forward_keys.append(forward_key)
        for reversed_syn in syn_list:
            if reversed_syn[0] == forward_key[1] and reversed_syn[1] == forward_key[0]:
                break
        if (syn[0], syn[1]) == forward_key:
            forward_exists = syn[-1]['exists']
            reciprocal_exists = reversed_syn[-1]['exists']
        else:
            reciprocal_exists = syn[-1]['exists']
            forward_exists = reversed_syn[-1]['exists']

        if forward_exists and reciprocal_exists:
            log_likelihood += np.log(dyads_distributions[forward_key][RECIPROCAL_DYAD_IDX])
        elif forward_exists and not reciprocal_exists:
            log_likelihood += np.log(dyads_distributions[forward_key][ONLY_UPPER_TRIANGLE_SYNAPSE_IDX])
        elif not forward_exists and reciprocal_exists:
            log_likelihood += np.log(dyads_distributions[forward_key][ONLY_LOWER_TRIANGLE_SYNAPSE_IDX])
        else:
            log_likelihood += np.log(dyads_distributions[forward_key][EMPTY_DYAD_IDX])
    return log_likelihood


def calc_reciprocal_dependence_model_log_likelihood_syn_list(spls, smi, beta, gamma, reference_age, developmental_ages,
                                                             syn_list_path):
    dyads_distributions = calc_reciprocal_dependence_model_dyads_states_distribution_syn_list(spls, smi, beta, gamma,
                                                                                              reference_age,
                                                                                              developmental_ages,
                                                                                              syn_list_path)
    log_likelihood = calc_reciprocal_dependence_model_log_likelihood_from_dyads_dist_string_keys(dyads_distributions,
                                                                                                 syn_list_path)
    return log_likelihood


def calc_reciprocal_dependence_model_average_adj_mat(spls, smi, beta, gamma, reference_age, developmental_ages,
                                                     c_elegans_connectome_path):
    dyads_distributions = calc_reciprocal_dependence_model_dyads_states_distribution(spls, smi, beta, gamma,
                                                                                     reference_age,
                                                                                     developmental_ages,
                                                                                     c_elegans_connectome_path)
    average_mat = calc_reciprocal_dependence_model_average_adj_mat_from_dyads_distributions(dyads_distributions,
                                                                                            c_elegans_connectome_path)
    return average_mat


def calc_reciprocal_dependence_model_average_adj_mat_from_dyads_distributions(dyads_distributions,
                                                                              num_neurons):
    average_adj_mat = np.zeros((num_neurons, num_neurons))
    for i in range(num_neurons - 1):
        for j in range(i + 1, num_neurons):
            average_adj_mat[i, j] = dyads_distributions[(i, j)][RECIPROCAL_DYAD_IDX] + dyads_distributions[(i, j)][
                ONLY_UPPER_TRIANGLE_SYNAPSE_IDX]
            average_adj_mat[j, i] = dyads_distributions[(i, j)][RECIPROCAL_DYAD_IDX] + dyads_distributions[(i, j)][
                ONLY_LOWER_TRIANGLE_SYNAPSE_IDX]
    return average_adj_mat


def calc_reciprocal_dependence_model_density_from_dyads_dist_index_keys(dyads_distributions, num_neurons):
    model_av_mat = calc_reciprocal_dependence_model_average_adj_mat_from_dyads_distributions(dyads_distributions,
                                                                                             num_neurons)
    mean_density = model_av_mat.sum() / (num_neurons * (num_neurons - 1))
    # contribution of variances of single synapses to the variance of the sum
    variance_num_synapses = (model_av_mat * (1 - model_av_mat)).sum()
    # contribution of the covariances of reciprocal synapses to the variance of the sum
    for neuronal_pair in dyads_distributions.keys():
        forth_prob = model_av_mat[neuronal_pair]
        back_prob = model_av_mat[neuronal_pair[1], neuronal_pair[0]]
        variance_num_synapses += 2 * (dyads_distributions[neuronal_pair][RECIPROCAL_DYAD_IDX] - forth_prob * back_prob)

    std_density = np.sqrt(variance_num_synapses) / (num_neurons * (num_neurons - 1))
    return mean_density, std_density


def calc_reciprocal_dependence_model_density_from_dyads_dist_string_keys(dyads_dist):
    model_density = 0
    model_density_std = 0
    for dyad in dyads_dist.keys():
        cur_dist = dyads_dist[dyad]
        forth_synapse_prob = cur_dist[RECIPROCAL_DYAD_IDX] + cur_dist[ONLY_UPPER_TRIANGLE_SYNAPSE_IDX]
        back_synapse_prob = cur_dist[RECIPROCAL_DYAD_IDX] + cur_dist[ONLY_LOWER_TRIANGLE_SYNAPSE_IDX]
        model_density += (forth_synapse_prob + back_synapse_prob)
        # The contribution of the variance of each one of the synapses
        model_density_std += (forth_synapse_prob * (1 - forth_synapse_prob) + back_synapse_prob * (
                1 - back_synapse_prob))
        # The contribution of the covariance of the synapses
        model_density_std += 2 * (
                cur_dist[RECIPROCAL_DYAD_IDX] - forth_synapse_prob * back_synapse_prob)
    model_density /= (2 * len(dyads_dist.keys()))
    model_density_std = np.sqrt(model_density_std) / (2 * len(dyads_dist.keys()))
    return model_density, model_density_std


def calc_reciprocal_dependence_model_reciprocity_from_dyads_dist(dyads_distributions, num_neurons):
    mean_num_reciprocal_dyads = 0
    variance_num_reciprocal_dyads = 0
    for i in range(num_neurons - 1):
        for j in range(i + 1, num_neurons):
            p = dyads_distributions[(i, j)][RECIPROCAL_DYAD_IDX]
            mean_num_reciprocal_dyads += p
            variance_num_reciprocal_dyads += p * (1 - p)
    possible_num_reciprocal_pairs = num_neurons * (num_neurons - 1) / 2
    mean_reciprocity = mean_num_reciprocal_dyads / possible_num_reciprocal_pairs
    std_reciprocity = np.sqrt(variance_num_reciprocal_dyads) / possible_num_reciprocal_pairs
    return mean_reciprocity, std_reciprocity


def calc_single_dyad_expected_flipped_synapses_between_runs(single_dyad_distribution):
    expected_num_flipped_synapses, variance_num_flipped_synapses = calc_single_dyad_expected_flipped_synapses_with_given_dyadic_state(
        single_dyad_distribution, single_dyad_distribution)
    return expected_num_flipped_synapses, variance_num_flipped_synapses


def calc_single_dyad_expected_flipped_synapses_with_given_dyadic_state(single_dyad_distribution, given_state):
    prob_for_2_flipped_synapses = 0
    prob_for_1_flipped_synapse = 0
    prob_for_0_flipped_synapses = 0
    # Iterating all 16 (4**2) possible options for 2 different dyad states
    prob_for_2_flipped_synapses += single_dyad_distribution[RECIPROCAL_DYAD_IDX] * given_state[EMPTY_DYAD_IDX]
    prob_for_2_flipped_synapses += single_dyad_distribution[ONLY_UPPER_TRIANGLE_SYNAPSE_IDX] * given_state[
        ONLY_LOWER_TRIANGLE_SYNAPSE_IDX]
    prob_for_2_flipped_synapses += single_dyad_distribution[ONLY_LOWER_TRIANGLE_SYNAPSE_IDX] * given_state[
        ONLY_UPPER_TRIANGLE_SYNAPSE_IDX]
    prob_for_2_flipped_synapses += single_dyad_distribution[EMPTY_DYAD_IDX] * given_state[RECIPROCAL_DYAD_IDX]
    prob_for_1_flipped_synapse += single_dyad_distribution[RECIPROCAL_DYAD_IDX] * given_state[
        ONLY_UPPER_TRIANGLE_SYNAPSE_IDX]
    prob_for_1_flipped_synapse += single_dyad_distribution[RECIPROCAL_DYAD_IDX] * given_state[
        ONLY_LOWER_TRIANGLE_SYNAPSE_IDX]
    prob_for_1_flipped_synapse += single_dyad_distribution[ONLY_UPPER_TRIANGLE_SYNAPSE_IDX] * given_state[
        RECIPROCAL_DYAD_IDX]
    prob_for_1_flipped_synapse += single_dyad_distribution[ONLY_UPPER_TRIANGLE_SYNAPSE_IDX] * given_state[
        EMPTY_DYAD_IDX]
    prob_for_1_flipped_synapse += single_dyad_distribution[ONLY_LOWER_TRIANGLE_SYNAPSE_IDX] * given_state[
        RECIPROCAL_DYAD_IDX]
    prob_for_1_flipped_synapse += single_dyad_distribution[ONLY_LOWER_TRIANGLE_SYNAPSE_IDX] * given_state[
        EMPTY_DYAD_IDX]
    prob_for_1_flipped_synapse += single_dyad_distribution[EMPTY_DYAD_IDX] * given_state[
        ONLY_UPPER_TRIANGLE_SYNAPSE_IDX]
    prob_for_1_flipped_synapse += single_dyad_distribution[EMPTY_DYAD_IDX] * given_state[
        ONLY_LOWER_TRIANGLE_SYNAPSE_IDX]
    prob_for_0_flipped_synapses += single_dyad_distribution[RECIPROCAL_DYAD_IDX] * given_state[RECIPROCAL_DYAD_IDX]
    prob_for_0_flipped_synapses += single_dyad_distribution[ONLY_UPPER_TRIANGLE_SYNAPSE_IDX] * given_state[
        ONLY_UPPER_TRIANGLE_SYNAPSE_IDX]
    prob_for_0_flipped_synapses += single_dyad_distribution[ONLY_LOWER_TRIANGLE_SYNAPSE_IDX] * given_state[
        ONLY_LOWER_TRIANGLE_SYNAPSE_IDX]
    prob_for_0_flipped_synapses += single_dyad_distribution[EMPTY_DYAD_IDX] * single_dyad_distribution[EMPTY_DYAD_IDX]

    expected_num_flipped_synapses = prob_for_1_flipped_synapse + 2 * prob_for_2_flipped_synapses
    # E[X**2]-E[X]**2
    variance_num_flipped_synapses = prob_for_1_flipped_synapse + 4 * prob_for_2_flipped_synapses - expected_num_flipped_synapses ** 2
    return expected_num_flipped_synapses, variance_num_flipped_synapses


def calc_reciprocal_dependence_model_variance_from_dyads_dist_str_keys(dyads_distribution):
    expected_num_flipped_synapses = 0
    variance_num_flipped_synapses = 0
    if len(dyads_distribution.keys()) == 0:
        return 0, 0
    for dyad in dyads_distribution.keys():
        cur_expected_num_flipped_synapses, cur_variace_num_flipped_synapses = calc_single_dyad_expected_flipped_synapses_between_runs(
            dyads_distribution[dyad])
        expected_num_flipped_synapses += cur_expected_num_flipped_synapses
        variance_num_flipped_synapses += cur_variace_num_flipped_synapses
    mean_expected_num_flipped_synapses = expected_num_flipped_synapses / (2 * len(dyads_distribution.keys()))
    mean_expected_num_flipped_synapses_std = np.sqrt(variance_num_flipped_synapses) / (
            2 * len(dyads_distribution.keys()))
    return mean_expected_num_flipped_synapses, mean_expected_num_flipped_synapses_std


def calc_reciprocal_dependence_model_data_cross_variance_from_dyads_dist_str_keys(dyads_distribution, syn_list_path):
    with open(syn_list_path, 'rb') as f:
        syn_list = pickle.load(f)
    expected_num_flipped_synapses = 0
    variance_num_flipped_synapses = 0
    for dyad in dyads_distribution.keys():
        data_state = [0] * 4
        num_assigned = 0
        for syn in syn_list:
            if (syn[0], syn[1]) == dyad:
                is_upper = syn[-1]['exists']
                num_assigned += 1
            if (syn[1], syn[0]) == dyad:
                is_lower = syn[-1]['exists']
                num_assigned += 1
            if num_assigned == 2:
                break
        if num_assigned < 2:
            raise ValueError("Invalid synapses list")
            return
        if is_upper and is_lower:
            data_state[RECIPROCAL_DYAD_IDX] = 1
        elif is_upper and not is_lower:
            data_state[ONLY_UPPER_TRIANGLE_SYNAPSE_IDX] = 1
        elif not is_upper and is_lower:
            data_state[ONLY_LOWER_TRIANGLE_SYNAPSE_IDX] = 1
        else:
            data_state[EMPTY_DYAD_IDX] = 1
        cur_expected_num_flipped_synapses, cur_variance_num_flipped_synapses = calc_single_dyad_expected_flipped_synapses_with_given_dyadic_state(
            dyads_distribution[dyad], data_state)
        expected_num_flipped_synapses += cur_expected_num_flipped_synapses
        variance_num_flipped_synapses += cur_variance_num_flipped_synapses

    mean_expected_num_flipped_synapses = expected_num_flipped_synapses / (2 * len(dyads_distribution.keys()))
    mean_expected_num_flipped_synapses_std = np.sqrt(variance_num_flipped_synapses) / (
            2 * len(dyads_distribution.keys()))
    return mean_expected_num_flipped_synapses, mean_expected_num_flipped_synapses_std


def calc_expected_number_of_synapses_of_type_reciprocal_dependence_model(neuronal_types_pair, smi, beta, gamma,
                                                                         reference_age,
                                                                         developmental_ages, spls, data_path):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    sorted_neurons = sorted(data.nodes)
    num_neurons = len(sorted_neurons)
    expected_num_synapses_forth = 0
    expected_num_synapses_back = 0
    spls_across_dev_forth = convert_spls_dict_to_spls_across_dev(spls, neuronal_types_pair[0], neuronal_types_pair[1])
    spls_across_dev_back = convert_spls_dict_to_spls_across_dev(spls, neuronal_types_pair[1], neuronal_types_pair[0])
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
            synapse_length = np.sqrt(
                np.sum((data.nodes[pre_neuron]['coords'] - data.nodes[post_neuron]['coords']) ** 2))
            dyad_state_distribution = calc_joint_reciprocal_synapse_distribution(
                spls_across_dev_forth, spls_across_dev_back, smi, beta, gamma,
                developmental_ages, synapse_age, synapse_birth_time, synapse_length)
            expected_num_synapses_forth += dyad_state_distribution[RECIPROCAL_DYAD_IDX] + dyad_state_distribution[
                ONLY_UPPER_TRIANGLE_SYNAPSE_IDX]
            expected_num_synapses_back += dyad_state_distribution[RECIPROCAL_DYAD_IDX] + dyad_state_distribution[
                ONLY_LOWER_TRIANGLE_SYNAPSE_IDX]
    return expected_num_synapses_forth, expected_num_synapses_back


def calc_expected_number_of_synapses_of_type_reciprocal_dependence_model_syn_list(neuronal_types_pair, smi, beta,
                                                                                  gamma, reference_age,
                                                                                  developmental_ages, spls,
                                                                                  syn_list):
    expected_num_synapses_forth = 0
    expected_num_synapses_back = 0
    spls_across_dev_forth = convert_spls_dict_to_spls_across_dev(spls, neuronal_types_pair[0], neuronal_types_pair[1])
    spls_across_dev_back = convert_spls_dict_to_spls_across_dev(spls, neuronal_types_pair[1], neuronal_types_pair[0])
    for synapse in syn_list:
        if neuronal_types_pair != synapse[-1]['type']:
            continue
        synapse_birth_time = synapse[-1]['birth time']
        if synapse_birth_time >= reference_age:
            continue
        synapse_age = reference_age - synapse_birth_time
        synapse_length = synapse[-1]['length']
        dyad_state_distribution = calc_joint_reciprocal_synapse_distribution(
            spls_across_dev_forth, spls_across_dev_back, smi, beta, gamma,
            developmental_ages, synapse_age, synapse_birth_time, synapse_length)
        expected_num_synapses_forth += dyad_state_distribution[RECIPROCAL_DYAD_IDX] + dyad_state_distribution[
            ONLY_UPPER_TRIANGLE_SYNAPSE_IDX]
        expected_num_synapses_back += dyad_state_distribution[RECIPROCAL_DYAD_IDX] + dyad_state_distribution[
            ONLY_LOWER_TRIANGLE_SYNAPSE_IDX]
    return expected_num_synapses_forth, expected_num_synapses_back


def single_binary_search_update(num_synapses_in_data, expected_num_synapses_model, cur_interval):
    if num_synapses_in_data > expected_num_synapses_model:
        cur_interval[0] = (cur_interval[0] + cur_interval[1]) / 2
    else:
        cur_interval[1] = (cur_interval[0] + cur_interval[1]) / 2
    return (cur_interval[0] + cur_interval[1]) / 2


def back_forth_alternating_binary_search_update(iteration, num_synapses_in_data_forth, expected_num_synapses_forth,
                                                num_synapses_in_data_back, expected_num_synapses_back, cur_square, spls,
                                                new_dev_stage, synaptic_type_forth, synaptic_type_back):
    if not iteration % 2:
        spls[new_dev_stage][synaptic_type_forth] = single_binary_search_update(num_synapses_in_data_forth,
                                                                               expected_num_synapses_forth,
                                                                               cur_square[0])
    else:
        spls[new_dev_stage][synaptic_type_back] = single_binary_search_update(num_synapses_in_data_back,
                                                                              expected_num_synapses_back, cur_square[1])


def back_forth_simultaneous_binary_search_update(num_synapses_in_data_forth, expected_num_synapses_forth,
                                                 num_synapses_in_data_back, expected_num_synapses_back, cur_square,
                                                 spls, new_dev_stage, synaptic_type_forth, synaptic_type_back):
    spls[new_dev_stage][synaptic_type_forth] = single_binary_search_update(num_synapses_in_data_forth,
                                                                           expected_num_synapses_forth,
                                                                           cur_square[0])
    spls[new_dev_stage][synaptic_type_back] = single_binary_search_update(num_synapses_in_data_back,
                                                                          expected_num_synapses_back, cur_square[1])


def search_spl_reciprocal_dependence_model(neuronal_types_pair, smi, beta, gamma, reference_age,
                                           developmental_ages, spls, type_configuration, data_path,
                                           tolerance=0.05):
    new_dev_stage = 0
    while reference_age > developmental_ages[new_dev_stage]:
        new_dev_stage += 1
    cur_square = np.array([[0.0, 1.0], [0.0, 1.0]])
    synaptic_type_forth = (neuronal_types_pair[0], neuronal_types_pair[1])
    synaptic_type_back = (neuronal_types_pair[1], neuronal_types_pair[0])
    spls[new_dev_stage][synaptic_type_forth] = (cur_square[0, 1] + cur_square[0, 0]) / 2
    spls[new_dev_stage][synaptic_type_back] = (cur_square[1, 1] + cur_square[1, 0]) / 2
    num_synapses_in_data_forth = count_synapses_of_type((neuronal_types_pair[0], neuronal_types_pair[1]),
                                                        type_configuration, data_path=data_path)
    num_synapses_in_data_back = count_synapses_of_type((neuronal_types_pair[1], neuronal_types_pair[0]),
                                                       type_configuration, data_path=data_path)
    expected_num_synapses_forth, expected_num_synapses_back = calc_expected_number_of_synapses_of_type_reciprocal_dependence_model(
        neuronal_types_pair, smi, beta, gamma, reference_age, developmental_ages, spls, data_path)

    # Perform binary search for the values of S+ that give on average the right amount of synapses as in the data.
    iteration = 0
    while np.fabs(num_synapses_in_data_forth - expected_num_synapses_forth) > tolerance or np.fabs(
            num_synapses_in_data_back - expected_num_synapses_back) > tolerance:
        if neuronal_types_pair[0] == neuronal_types_pair[1]:
            back_forth_simultaneous_binary_search_update(num_synapses_in_data_forth, expected_num_synapses_forth,
                                                         num_synapses_in_data_back, expected_num_synapses_back,
                                                         cur_square, spls, new_dev_stage, synaptic_type_forth,
                                                         synaptic_type_back)
        else:
            back_forth_alternating_binary_search_update(iteration, num_synapses_in_data_forth,
                                                        expected_num_synapses_forth, num_synapses_in_data_back,
                                                        expected_num_synapses_back, cur_square, spls, new_dev_stage,
                                                        synaptic_type_forth, synaptic_type_back)

        expected_num_synapses_forth, expected_num_synapses_back = calc_expected_number_of_synapses_of_type_reciprocal_dependence_model(
            neuronal_types_pair, smi, beta, gamma, reference_age, developmental_ages, spls, data_path)

        if iteration > 2 * 10 * np.log(10) / np.log(2):
            # Probably the number of synapses can't be reached up to the tolerance with the given parameters, as the
            # size of the square side is already smaller than 10e-10.
            break

        iteration += 1
    return spls[new_dev_stage][synaptic_type_forth], spls[new_dev_stage][synaptic_type_back]


def search_spl_reciprocal_dependence_model_syn_list(neuronal_types_pair, smi, beta, gamma, reference_age,
                                                    developmental_ages, spls, synapses_list, tolerance=0.05):
    new_dev_stage = 0
    while reference_age > developmental_ages[new_dev_stage]:
        new_dev_stage += 1
    cur_square = np.array([[0.0, 1.0], [0.0, 1.0]])
    synaptic_type_forth = (neuronal_types_pair[0], neuronal_types_pair[1])
    synaptic_type_back = (neuronal_types_pair[1], neuronal_types_pair[0])
    spls[new_dev_stage][synaptic_type_forth] = (cur_square[0, 1] + cur_square[0, 0]) / 2
    spls[new_dev_stage][synaptic_type_back] = (cur_square[1, 1] + cur_square[1, 0]) / 2
    existing_synapses_of_forth_type_in_data = len(
        [synapse for synapse in synapses_list if
         synapse[-1]["type"] == synaptic_type_forth and synapse[-1]["exists"]])
    existing_synapses_of_back_type_in_data = len(
        [synapse for synapse in synapses_list if
         synapse[-1]["type"] == synaptic_type_back and synapse[-1]['exists']])
    expected_num_synapses_forth, expected_num_synapses_back = calc_expected_number_of_synapses_of_type_reciprocal_dependence_model_syn_list(
        neuronal_types_pair, smi, beta, gamma, reference_age, developmental_ages, spls, synapses_list)

    # Perform binary search for the values of S+ that give on average the right amount of synapses as in the data.
    iteration = 0
    while np.fabs(existing_synapses_of_forth_type_in_data - expected_num_synapses_forth) > tolerance or np.fabs(
            existing_synapses_of_back_type_in_data - expected_num_synapses_back) > tolerance:
        if neuronal_types_pair[0] == neuronal_types_pair[1]:
            back_forth_simultaneous_binary_search_update(existing_synapses_of_forth_type_in_data,
                                                         expected_num_synapses_forth,
                                                         existing_synapses_of_back_type_in_data,
                                                         expected_num_synapses_back,
                                                         cur_square, spls, new_dev_stage, synaptic_type_forth,
                                                         synaptic_type_back)
        else:
            back_forth_alternating_binary_search_update(iteration, existing_synapses_of_forth_type_in_data,
                                                        expected_num_synapses_forth,
                                                        existing_synapses_of_back_type_in_data,
                                                        expected_num_synapses_back, cur_square, spls, new_dev_stage,
                                                        synaptic_type_forth, synaptic_type_back)

        expected_num_synapses_forth, expected_num_synapses_back = calc_expected_number_of_synapses_of_type_reciprocal_dependence_model_syn_list(
            neuronal_types_pair, smi, beta, gamma, reference_age, developmental_ages, spls, synapses_list)

        if iteration > 2 * 10 * np.log(10) / np.log(2):
            # Probably the number of synapses can't be reached up to the tolerance with the given parameters, as the
            # size of the square side is already smaller than 10e-10.
            break

        iteration += 1
    return spls[new_dev_stage][synaptic_type_forth], spls[new_dev_stage][synaptic_type_back]


def sample_from_dyads_distribution(dyads_distributions_path):
    with open(dyads_distributions_path, 'rb') as f:
        dyads_distribution = pickle.load(f)
    adj_mat = np.zeros((MEI_ZHEN_NUM_NEURONS, MEI_ZHEN_NUM_NEURONS))
    for dyad in dyads_distribution.keys():
        p = np.random.rand()
        if p < dyads_distribution[dyad][RECIPROCAL_DYAD_IDX]:
            adj_mat[dyad[0], dyad[1]] = 1
            adj_mat[dyad[1], dyad[0]] = 1
        elif p < dyads_distribution[dyad][RECIPROCAL_DYAD_IDX] + dyads_distribution[dyad][
            ONLY_UPPER_TRIANGLE_SYNAPSE_IDX]:
            adj_mat[dyad[0], dyad[1]] = 1
        elif p < dyads_distribution[dyad][RECIPROCAL_DYAD_IDX] + dyads_distribution[dyad][
            ONLY_UPPER_TRIANGLE_SYNAPSE_IDX] + dyads_distribution[dyad][ONLY_LOWER_TRIANGLE_SYNAPSE_IDX]:
            adj_mat[dyad[1], dyad[0]] = 1
    return adj_mat


def sample_from_dyads_distribution_train_test(train_dyads_dists_path, test_dyads_dists_path, neurons_list):
    with open(train_dyads_dists_path, 'rb') as f:
        train_dyads = pickle.load(f)
    with open(test_dyads_dists_path, 'rb') as f:
        test_dyads = pickle.load(f)
    num_neurons = len(neurons_list)
    adj_mat = np.zeros((num_neurons, num_neurons))
    for n1_idx in range(num_neurons - 1):
        for n2_idx in range(n1_idx + 1, num_neurons):
            forward_key = tuple(sorted([neurons_list[n1_idx], neurons_list[n2_idx]]))
            cur_dyads = train_dyads if forward_key in train_dyads.keys() else test_dyads
            pre_idx = neurons_list.index(forward_key[0])
            post_idx = neurons_list.index(forward_key[1])
            p = np.random.rand()
            if p < cur_dyads[forward_key][RECIPROCAL_DYAD_IDX]:
                adj_mat[pre_idx, post_idx] = 1
                adj_mat[post_idx, pre_idx] = 1
            elif p < cur_dyads[forward_key][RECIPROCAL_DYAD_IDX] + cur_dyads[forward_key][
                ONLY_UPPER_TRIANGLE_SYNAPSE_IDX]:
                adj_mat[pre_idx, post_idx] = 1
            elif p < cur_dyads[forward_key][RECIPROCAL_DYAD_IDX] + cur_dyads[forward_key][
                ONLY_UPPER_TRIANGLE_SYNAPSE_IDX] + cur_dyads[forward_key][ONLY_LOWER_TRIANGLE_SYNAPSE_IDX]:
                adj_mat[post_idx, pre_idx] = 1
    return adj_mat


def _save_max_like_params_per_split(likelihoods_path, out_path):
    max_like_vals_per_split = {}
    for split in os.listdir(likelihoods_path):
        smi, beta, _ = find_max_likelihood_full_model(os.path.join(likelihoods_path, split))
        max_like_vals_per_split[split] = {'S-': smi, 'beta': beta}
    with open(out_path, 'wb') as f:
        pickle.dump(max_like_vals_per_split, f)


def save_max_like_params_per_split_single_epoch():
    likelihoods_path = f'SavedOutputs\ReciprocalModel\\DyadsSplit\likelihoods\\SingleDevStage'
    out_path = f'SavedOutputs\ReciprocalModel\\DyadsSplit\\max_likelihood_params_per_split_single_epoch.pkl'
    _save_max_like_params_per_split(likelihoods_path, out_path)


def save_max_like_params_per_split_three_epochs():
    likelihoods_path = f'SavedOutputs\ReciprocalModel\\DyadsSplit\likelihoods\\ThreeDevStages'
    out_path = f'SavedOutputs\ReciprocalModel\\DyadsSplit\\max_likelihood_params_per_split_3_epochs.pkl'
    _save_max_like_params_per_split(likelihoods_path, out_path)


def construct_probs_array_from_dyads_dist_string_keys(dyads_dist):
    syn_probs = []
    for dyad in dyads_dist.keys():
        cur_dist = dyads_dist[dyad]
        forth_synapse_prob = cur_dist[RECIPROCAL_DYAD_IDX] + cur_dist[ONLY_UPPER_TRIANGLE_SYNAPSE_IDX]
        back_synapse_prob = cur_dist[RECIPROCAL_DYAD_IDX] + cur_dist[ONLY_LOWER_TRIANGLE_SYNAPSE_IDX]
        syn_probs.append((dyad[0], dyad[1], forth_synapse_prob))
        syn_probs.append((dyad[1], dyad[0], back_synapse_prob))
    syn_probs = sorted(syn_probs)
    only_probs = np.zeros(len(syn_probs))
    i = 0
    for syn in syn_probs:
        only_probs[i] = syn[-1]
        i += 1
    return only_probs
