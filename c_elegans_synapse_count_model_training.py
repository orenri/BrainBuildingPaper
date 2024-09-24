import numpy as np
import pickle
from copy import deepcopy

from c_elegans_independent_model_training import calc_elongation_factor, convert_spls_dict_to_spls_across_dev, \
    count_synapses_of_type
from c_elegans_reciprocal_model_training import single_binary_search_update


def p_synapse_is_formed(spls_across_dev, smi, beta, developmental_ages, time_passed_from_given_state,
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

            # The probability to have no synapses is the probability of having no synapses at the previous stage, and
            # not forming any now.
            elif syn_count == 0:
                num_synapses_pmf_across_time[step, syn_count] = num_synapses_pmf_across_time[step - 1, syn_count] * (
                        1 - smi)

            # Handel the edge case of the maximal number of synapses - there is no entry holding the probability for
            # more synapses in the matrix. The probability is to have 1 less at the previous step, to form now, and to
            # not erase anything.
            elif syn_count == max_num_synapses_at_the_end:
                num_synapses_pmf_across_time[step, syn_count] = (
                        num_synapses_pmf_across_time[step - 1, syn_count - 1] * formation_prob * (1 - smi))

            else:
                # Don't prune and don't form
                prob_to_stay_the_same = num_synapses_pmf_across_time[step - 1, syn_count] * (1 - smi) * (
                        1 - formation_prob)

                # Form a connection
                prob_to_add_connection = num_synapses_pmf_across_time[step - 1, syn_count - 1] * formation_prob
                # If there were connections before, don't prune them
                if syn_count - 1 > 0:
                    prob_to_add_connection *= (1 - smi)

                prob_to_remove_connection = num_synapses_pmf_across_time[step - 1, syn_count + 1] * smi * (
                        1 - formation_prob)

                num_synapses_pmf_across_time[
                    step, syn_count] = prob_to_stay_the_same + prob_to_add_connection + prob_to_remove_connection

    # The probability to have at least one synapse at the final step
    return num_synapses_pmf_across_time[-1, 1:].sum()


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

            expected_num_synapses += p_synapse_is_formed(spls_across_dev, smi, beta, developmental_ages, synapse_age,
                                                         reference_age, norm_dist_between_neurons)

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
    iter = 0
    while abs(expected_num_synapses_model - num_synapses_in_data) > tolerance:
        spls_copy[new_dev_stage][neuronal_types_pair] = single_binary_search_update(num_synapses_in_data,
                                                                                    expected_num_synapses_model,
                                                                                    cur_search_window)
        expected_num_synapses_model = calc_expected_num_synapses_type_pair(neuronal_types_pair, smi, beta,
                                                                           reference_age,
                                                                           developmental_ages, spls_copy, data_path)
        if iter > 2 * 10 * np.log(10) / np.log(2):
            # Probably the number of synapses can't be reached up to the tolerance with the given parameters, as the
            # size of the square side is already smaller than 10e-10.
            break
        iter += 1
    return spls_copy[new_dev_stage][neuronal_types_pair]
