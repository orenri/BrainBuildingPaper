import os
import pickle
import numpy as np
import networkx as nx
from CElegansNeuronsAdder import CElegansNeuronsAdder
from c_elegans_independent_model_training import count_synapses_of_type, count_neurons_of_type
from c_elegans_constants import ADULT_WORM_AGE
from c_elegans_data_parsing import SYNAPTIC_COARSE_TYPES


# Generates a graph built of multiple ER graphs, each one for a sub graph of all the synapses of the same type in the C.
# elegans connectome, using p that is equal to (# formed synapses of type) / (# possible synapses of type)
def generate_er_block_per_type(type_configuration, measured_connectome_path, neuron_subset_path,
                               c_elegans_data_path=CElegansNeuronsAdder.DEFAULT_DATA_PATH, num_artificial_types=1,
                               types_file_name=CElegansNeuronsAdder.DEFAULT_TYPES_FILE_NAME):
    if CElegansNeuronsAdder.SINGLE_TYPE == type_configuration:
        synaptic_types_dict = {(None, None): 1}
    elif CElegansNeuronsAdder.COARSE_TYPES == type_configuration:
        synaptic_types_dict = SYNAPTIC_COARSE_TYPES.copy()
    elif CElegansNeuronsAdder.SUB_TYPES == type_configuration:
        with open(os.path.join(os.getcwd(), "CElegansData", "SubTypes", "synaptic_subtypes_dict.pkl"), 'rb') as f:
            synaptic_types_dict = pickle.load(f)
    elif CElegansNeuronsAdder.ARTIFICIAL_TYPES == type_configuration:
        counter = 0
        synaptic_types_dict = {}
        for i in range(num_artificial_types):
            for j in range(num_artificial_types):
                synaptic_types_dict[(i, j)] = counter
                counter += 1
    with open(os.path.join(c_elegans_data_path, measured_connectome_path), 'rb') as f:
        measured_connectome = pickle.load(f)

    for syn_type in synaptic_types_dict.keys():
        formed_synapses = count_synapses_of_type(syn_type, type_configuration, connectome=measured_connectome)
        num_pre = count_neurons_of_type(syn_type[0], ADULT_WORM_AGE, neuron_subset_path, type_configuration,
                                        c_elegans_data_path=c_elegans_data_path, types_file_name=types_file_name)
        if syn_type[0] == syn_type[1]:
            num_possible_synapses = num_pre * (num_pre - 1)
        else:
            num_post = count_neurons_of_type(syn_type[1], ADULT_WORM_AGE, neuron_subset_path, type_configuration,
                                             c_elegans_data_path=c_elegans_data_path, types_file_name=types_file_name)
            num_possible_synapses = num_pre * num_post
        if num_possible_synapses == 0:
            synaptic_types_dict[syn_type] = 0
            continue
        p = formed_synapses / num_possible_synapses
        synaptic_types_dict[syn_type] = p
    # Form an empty graph with all neurons
    combined_er = measured_connectome.copy()
    edges_to_remove = [edge for edge in combined_er.edges()]
    combined_er.remove_edges_from(edges_to_remove)
    # Generate a sample graph
    for n1 in combined_er.nodes:
        for n2 in combined_er.nodes:
            if n1 == n2:
                continue
            r = np.random.rand()
            if r <= synaptic_types_dict[(combined_er.nodes[n1]["type"], combined_er.nodes[n2]["type"])]:
                length = np.sqrt(np.sum((combined_er.nodes[n1]["coords"] - combined_er.nodes[n2]["coords"]) ** 2))
                birth_time = max(combined_er.nodes[n1]["birth_time"], combined_er.nodes[n2]["birth_time"])
                combined_er.add_edges_from([(n1, n2, {'length': length, 'birth time': birth_time})])
    # Generate average adjacency matrix (each entry is the probability for the corresponding synapse to be formed)
    sorted_neurons = sorted(list(combined_er.nodes))
    num_neurons = len(sorted_neurons)
    av_adj_mat = np.zeros((num_neurons, num_neurons))

    for i in range(num_neurons - 1):
        for j in range(i + 1, num_neurons):
            av_adj_mat[i, j] = synaptic_types_dict[
                (combined_er.nodes[sorted_neurons[i]]["type"], combined_er.nodes[sorted_neurons[j]]["type"])]
            av_adj_mat[j, i] = synaptic_types_dict[
                (combined_er.nodes[sorted_neurons[j]]["type"], combined_er.nodes[sorted_neurons[i]]["type"])]

    return combined_er, av_adj_mat


# Calculates the expected value of number of mutual synapses of a model that randomly samples the right amount of
# synapses of each type (the same number as appears in data).
def calculate_total_chance_level_hit_rate(connectome_path=None, connectome=None):
    if connectome_path is not None:
        with open(connectome_path, 'rb') as f:
            connectome = pickle.load(f)
    number_of_synapses_per_type = {}
    number_of_neurons_per_type = {}
    for syn in connectome.edges():
        syn_type = (connectome.nodes[syn[0]]["type"], connectome.nodes[syn[1]]["type"])
        if syn_type not in number_of_synapses_per_type.keys():
            number_of_synapses_per_type[syn_type] = 1
        else:
            number_of_synapses_per_type[syn_type] += 1
        if syn_type[0] not in number_of_neurons_per_type.keys():
            neurons_of_desired_type = [neuron for neuron in connectome.nodes if
                                       connectome.nodes[neuron]["type"] == syn_type[0]]
            num_neurons = len(neurons_of_desired_type)
            number_of_neurons_per_type[syn_type[0]] = num_neurons
        if syn_type[1] not in number_of_neurons_per_type.keys():
            neurons_of_desired_type = [neuron for neuron in connectome.nodes if
                                       connectome.nodes[neuron]["type"] == syn_type[1]]
            num_neurons = len(neurons_of_desired_type)
            number_of_neurons_per_type[syn_type[1]] = num_neurons

    total_chance_level_hit = 0
    for syn_type in number_of_synapses_per_type.keys():
        if syn_type[0] == syn_type[1]:
            num_possible_synapses = number_of_neurons_per_type[syn_type[0]] * (
                    number_of_neurons_per_type[syn_type[0]] - 1)
        else:
            num_possible_synapses = number_of_neurons_per_type[syn_type[0]] * number_of_neurons_per_type[syn_type[1]]
        total_chance_level_hit += number_of_synapses_per_type[syn_type] ** 2 / num_possible_synapses
    total_num_synapses = len(list(connectome.edges))
    total_chance_level_hit_rate = total_chance_level_hit / total_num_synapses
    return total_chance_level_hit_rate


# Calculates the chance level hit rate (mutual synapses between neurons of type1 and of type2 with the given adj_mat)
# when sampling randomly the correct amount of synapses of the given type.
# The indexing of the neurons corresponds to their row/column in the matrix and the types are sets of indices referred
# to as sharing a type.
def calc_chance_level_hit_specific_arbitrary_type(adj_mat, type1, type2):
    num_neurons_of_type1 = len(type1)
    num_neurons_of_type2 = len(type2)
    if type1 == type2:
        num_possible_synapses = num_neurons_of_type1 * (num_neurons_of_type1 - 1)
    else:
        num_possible_synapses = num_neurons_of_type1 * num_neurons_of_type2

    if num_possible_synapses == 0:
        return 0
    num_existing_synapses = 0
    for n1 in type1:
        for n2 in type2:
            if adj_mat[n1, n2] == 1:
                num_existing_synapses += 1
    return num_existing_synapses ** 2 / num_possible_synapses


# Calculates the total chance level hit rate corresponding to the given split into types.
# types is a list of sets, eah contains indices of neurons that are referred to as sharing a type.
def calc_chance_level_hit_rate_arbitrary_types(adj_mat, types):
    total_hit = 0
    for type1 in types:
        for type2 in types:
            total_hit += calc_chance_level_hit_specific_arbitrary_type(adj_mat, type1, type2)
    total_hit_rate = total_hit / adj_mat.sum()
    return total_hit_rate


# Greedily clusters neurons into arbitrary types such that the total hit rate for Block ER model using these types is
# maximised (each iteration unions 2 types into one such that the loss in hit rate is minimized).
def greedy_type_aggregation(adj_mat, out_path, initial_types=None):
    if initial_types is None:
        types = []
        num_neurons = adj_mat.shape[0]
        for i in range(num_neurons):
            types.append({i})
    else:
        types = initial_types
    while len(types) > 1:
        hit_rate = calc_chance_level_hit_rate_arbitrary_types(adj_mat, types)
        with open(os.path.join(out_path, f"{len(types)}.pkl"), 'wb') as f:
            pickle.dump((hit_rate, types), f)
        cur_min_hit_rate_decrease = 1
        for idx1 in range(len(types) - 1):
            for idx2 in range(idx1 + 1, len(types)):
                cur_merged_types = types.copy()
                cur_merged_types[idx1] = cur_merged_types[idx1].union(cur_merged_types[idx2])
                cur_merged_types.remove(cur_merged_types[idx2])
                cur_merged_types_hit_rate = calc_chance_level_hit_rate_arbitrary_types(adj_mat, cur_merged_types)
                cur_hit_rate_decrease = hit_rate - cur_merged_types_hit_rate
                if cur_hit_rate_decrease < cur_min_hit_rate_decrease:
                    cur_min_hit_rate_decrease = cur_hit_rate_decrease
                    cur_min_merged_types = cur_merged_types
        types = cur_min_merged_types


def random_types_aggregation(adj_mat, out_path):
    num_neurons = adj_mat.shape[0]
    types = [{i} for i in range(num_neurons)]
    while len(types) > 0:
        cur_num_types = len(types)
        hit_rate = calc_chance_level_hit_rate_arbitrary_types(adj_mat, types)
        with open(os.path.join(out_path, f"{cur_num_types}.pkl"), 'wb') as f:
            pickle.dump((hit_rate, types), f)
        if cur_num_types == 1:
            break
        cur_types_to_merge = np.random.randint(low=0, high=cur_num_types, size=2)
        while cur_types_to_merge[0] == cur_types_to_merge[1]:
            cur_types_to_merge = np.random.randint(low=0, high=cur_num_types, size=2)
        types_copy = types.copy()
        types[cur_types_to_merge[0]] = types_copy[cur_types_to_merge[0]].union(types_copy[cur_types_to_merge[1]])
        types.remove(types_copy[cur_types_to_merge[1]])


def generate_er_single_type(measured_connectome_path):
    with open(measured_connectome_path, 'rb') as f:
        measured_connectome = pickle.load(f)
    density = nx.density(measured_connectome)
    generated_connectome = measured_connectome.copy()
    edges_to_remove = [edge for edge in generated_connectome.edges()]
    generated_connectome.remove_edges_from(edges_to_remove)
    # Generate a sample graph
    for n1 in generated_connectome.nodes:
        for n2 in generated_connectome.nodes:
            if n1 == n2:
                continue
            r = np.random.rand()
            if r <= density:
                length = np.sqrt(
                    np.sum((generated_connectome.nodes[n1]["coords"] - generated_connectome.nodes[n2]["coords"]) ** 2))
                birth_time = max(generated_connectome.nodes[n1]["birth_time"],
                                 generated_connectome.nodes[n2]["birth_time"])
                generated_connectome.add_edges_from([(n1, n2, {'length': length, 'birth time': birth_time})])
    return generated_connectome
