import networkx as nx
import pandas as pd
import numpy as np
import pickle
import os
from copy import deepcopy
from CElegansNeuronsAdder import CElegansNeuronsAdder
from c_elegans_constants import WORM_LENGTH_NORMALIZATION, ADULT_WORM_AGE, FULL_DEVELOPMENTAL_AGES

# Connectivity types
N1_TO_N2 = ['S', 'Sp']
N2_TO_N1 = ['R', 'Rp']
RECIPROCAL = ['EJ']

# Column names in the connectivity excel file.
TYPE_COL = "Type"
NEURON_1_COL = "Neuron 1"
NEURON_2_COL = "Neuron 2"

DEFAULT_DATA_PATH = CElegansNeuronsAdder.DEFAULT_DATA_PATH
DEFAULT_BIRTH_TIMES_FILE_NAME = CElegansNeuronsAdder.DEFAULT_BIRTH_TIMES_FILE_NAME
DEFAULT_POSITIONS_FILE_NAME_1D = CElegansNeuronsAdder.DEFAULT_POSITIONS_FILE_NAME_1D
DEFAULT_POSITIONS_FILE_NAME_3D = CElegansNeuronsAdder.DEFAULT_POSITIONS_FILE_NAME_3D
DEFAULT_TYPES_FILE_NAME = CElegansNeuronsAdder.DEFAULT_TYPES_FILE_NAME

# Columns and strings in the xls files
POSITION_COL = CElegansNeuronsAdder.POSITION_COL
BIRTH_TIME_COL = CElegansNeuronsAdder.BIRTH_TIME_COL
CELL_TYPE_COL = CElegansNeuronsAdder.CELL_TYPE_COL
CELL_SUBTYPE_COL = CElegansNeuronsAdder.CELL_SUBTYPE_COL

# Neuronal types in cell types excel
SENSORY_XLS = CElegansNeuronsAdder.SENSORY_XLS
MOTOR_XLS = CElegansNeuronsAdder.MOTOR_XLS
INTER_XLS = CElegansNeuronsAdder.INTER_XLS

POSITION_STR = CElegansNeuronsAdder.POSITION_STR
BIRTH_TIME_STR = CElegansNeuronsAdder.BIRTH_TIME_STR  # in minutes
TYPE_STR = CElegansNeuronsAdder.TYPE_STR

# Neuronal subtypes in cell types excel
SENSORY1_XLS = CElegansNeuronsAdder.SENSORY1_XLS
SENSORY2_XLS = CElegansNeuronsAdder.SENSORY2_XLS
SENSORY3_XLS = CElegansNeuronsAdder.SENSORY3_XLS
SENSORY4_XLS = CElegansNeuronsAdder.SENSORY4_XLS
SENSORY5_XLS = CElegansNeuronsAdder.SENSORY5_XLS
SENSORY6_XLS = CElegansNeuronsAdder.SENSORY6_XLS
MOTOR1_XLS = CElegansNeuronsAdder.MOTOR1_XLS
MOTOR2_XLS = CElegansNeuronsAdder.MOTOR2_XLS
MOTOR3_XLS = CElegansNeuronsAdder.MOTOR3_XLS
MOTOR4_XLS = CElegansNeuronsAdder.MOTOR4_XLS
INTER1_XLS = CElegansNeuronsAdder.INTER1_XLS
INTER2_XLS = CElegansNeuronsAdder.INTER2_XLS
INTER3_XLS = CElegansNeuronsAdder.INTER3_XLS
INTER4_XLS = CElegansNeuronsAdder.INTER4_XLS
INTER5_XLS = CElegansNeuronsAdder.INTER5_XLS

# Neuronal types in connectome
SENSORY = CElegansNeuronsAdder.SENSORY
SENSORY1 = CElegansNeuronsAdder.SENSORY1
SENSORY2 = CElegansNeuronsAdder.SENSORY2
SENSORY3 = CElegansNeuronsAdder.SENSORY3
SENSORY4 = CElegansNeuronsAdder.SENSORY4
SENSORY5 = CElegansNeuronsAdder.SENSORY5
SENSORY6 = CElegansNeuronsAdder.SENSORY6
SENSORY_TYPES = [SENSORY, SENSORY1, SENSORY2, SENSORY3, SENSORY4, SENSORY5, SENSORY6]
MOTOR = CElegansNeuronsAdder.MOTOR
MOTOR1 = CElegansNeuronsAdder.MOTOR1
MOTOR2 = CElegansNeuronsAdder.MOTOR2
MOTOR3 = CElegansNeuronsAdder.MOTOR3
MOTOR4 = CElegansNeuronsAdder.MOTOR4
MOTOR_TYPES = [MOTOR, MOTOR1, MOTOR2, MOTOR3, MOTOR4]
INTER = CElegansNeuronsAdder.INTER
INTER1 = CElegansNeuronsAdder.INTER1
INTER2 = CElegansNeuronsAdder.INTER2
INTER3 = CElegansNeuronsAdder.INTER3
INTER4 = CElegansNeuronsAdder.INTER4
INTER5 = CElegansNeuronsAdder.INTER5
INTER_TYPES = [INTER, INTER1, INTER2, INTER3, INTER4, INTER5]

# Mapping between excel file subtypes and code subtypes
CELL_TYPE_MAPPING = CElegansNeuronsAdder.CELL_TYPE_MAPPING

# Neuronal subtypes mapping to course types
SUB_COARSE_TYPES_MAPPING = {SENSORY: SENSORY, SENSORY1: SENSORY, SENSORY2: SENSORY, SENSORY3: SENSORY,
                            SENSORY4: SENSORY, SENSORY5: SENSORY, SENSORY6: SENSORY, MOTOR: MOTOR, MOTOR1: MOTOR,
                            MOTOR2: MOTOR, MOTOR3: MOTOR, MOTOR4: MOTOR, INTER: INTER, INTER1: INTER, INTER2: INTER,
                            INTER3: INTER, INTER4: INTER, INTER5: INTER}

# A list of all the possible neuronal subtypes.
NEURONAL_SUBTYPES = [SENSORY1, SENSORY2, SENSORY3, SENSORY4, SENSORY5, SENSORY6, MOTOR1, MOTOR2, MOTOR3,
                     MOTOR4, INTER1, INTER2, INTER3, INTER4, INTER5]

# Synapses types
SYNAPTIC_COARSE_TYPES = {(SENSORY, SENSORY): 0, (SENSORY, MOTOR): 1, (SENSORY, INTER): 2, (MOTOR, SENSORY): 3,
                         (MOTOR, MOTOR): 4, (MOTOR, INTER): 5, (INTER, SENSORY): 6, (INTER, MOTOR): 7,
                         (INTER, INTER): 8}

# Rows and columns names in Mei Jhen's data sets
ACROSS_DEVELOPMENT_DATA_SETS = [f"Dataset{i}" for i in range(1, 9)]
PRE_SYNAPTIC = "pre"
POST_SYNAPTIC = "post"

# Neuronal class column in the biological neuronal classes dataset
CLASS_COL = 'Neuron class'


def parse_connectivity_worm_atlas(connectivity_path, positions_path, birth_times_path, type_path, do_include_gap_jns,
                                  out_path="CElegansData\\worm_atlas_connectome_chemical_no_autosynapses_subtypes.pkl"):
    connectome_df = pd.read_excel(connectivity_path)
    positions = pd.read_excel(positions_path, index_col=0)
    birth_times = pd.read_excel(birth_times_path, index_col=0)
    cell_types = pd.read_excel(type_path, index_col=0, sheet_name=None)
    connectome = nx.empty_graph(0, create_using=nx.DiGraph())
    for index, row in connectome_df.iterrows():
        # Ignore auto-synapses
        if connectome_df[NEURON_1_COL][index] == connectome_df[NEURON_2_COL][index]:
            continue
        # Decide which edge should be added
        connection_type = connectome_df[TYPE_COL][index]
        if not (connection_type in N1_TO_N2 or connection_type in N2_TO_N1 or (
                connection_type in RECIPROCAL and do_include_gap_jns)):
            continue
        else:
            length = np.abs(positions.loc[connectome_df[NEURON_1_COL][index], POSITION_COL] - positions.loc[
                connectome_df[NEURON_2_COL][index], POSITION_COL])
            birth_time = max(birth_times.loc[connectome_df[NEURON_1_COL][index], BIRTH_TIME_COL],
                             birth_times.loc[connectome_df[NEURON_2_COL][index], BIRTH_TIME_COL])
            if connection_type in N1_TO_N2:
                edges = [(connectome_df[NEURON_1_COL][index], connectome_df[NEURON_2_COL][index],
                          {'length': length, 'birth time': birth_time})]
            elif connection_type in N2_TO_N1:
                edges = [(connectome_df[NEURON_2_COL][index], connectome_df[NEURON_1_COL][index],
                          {'length': length, 'birth time': birth_time})]
            elif connection_type in RECIPROCAL:
                edges = [(connectome_df[NEURON_1_COL][index], connectome_df[NEURON_2_COL][index],
                          {'length': length, 'birth time': birth_time}),
                         (connectome_df[NEURON_2_COL][index], connectome_df[NEURON_1_COL][index],
                          {'length': length, 'birth time': birth_time})]

            # Validate that the nodes already exist in the graph
            if connectome_df[NEURON_1_COL][index] not in connectome.nodes():
                for sheet in cell_types.keys():
                    if connectome_df[NEURON_1_COL][index] in cell_types[sheet].index:
                        neuron_type_xls = cell_types[sheet].loc[connectome_df[NEURON_1_COL][index], CELL_TYPE_COL]
                        neurons_subtype_xls = cell_types[sheet].loc[
                            connectome_df[NEURON_1_COL][index], CELL_SUBTYPE_COL]
                        if neurons_subtype_xls in CELL_TYPE_MAPPING.keys():
                            neuron_type = CELL_TYPE_MAPPING[neurons_subtype_xls]
                        else:
                            if neuron_type_xls in SENSORY_XLS:
                                neuron_type = SENSORY
                            elif neuron_type_xls in MOTOR_XLS:
                                neuron_type = MOTOR
                            elif neuron_type_xls in INTER_XLS:
                                neuron_type = INTER
                        break

                connectome.add_node(connectome_df[NEURON_1_COL][index],
                                    coords=positions.loc[connectome_df[NEURON_1_COL][index], POSITION_COL],
                                    birth_time=birth_times.loc[connectome_df[NEURON_1_COL][index], BIRTH_TIME_COL],
                                    type=neuron_type)
            if connectome_df[NEURON_2_COL][index] not in connectome.nodes():
                for sheet in cell_types.keys():
                    if connectome_df[NEURON_2_COL][index] in cell_types[sheet].index:
                        neuron_type_xls = cell_types[sheet].loc[connectome_df[NEURON_2_COL][index], CELL_TYPE_COL]
                        neurons_subtype_xls = cell_types[sheet].loc[
                            connectome_df[NEURON_2_COL][index], CELL_SUBTYPE_COL]
                        if neurons_subtype_xls in CELL_TYPE_MAPPING.keys():
                            neuron_type = CELL_TYPE_MAPPING[neurons_subtype_xls]
                        else:
                            if neuron_type_xls in SENSORY_XLS:
                                neuron_type = SENSORY
                            elif neuron_type_xls in MOTOR_XLS:
                                neuron_type = MOTOR
                            elif neuron_type_xls in INTER_XLS:
                                neuron_type = INTER
                        break
                connectome.add_node(connectome_df[NEURON_2_COL][index],
                                    coords=positions.loc[connectome_df[NEURON_2_COL][index], POSITION_COL],
                                    birth_time=birth_times.loc[connectome_df[NEURON_2_COL][index], BIRTH_TIME_COL],
                                    type=neuron_type)

            # Add the edges (if they already exist their data is updated, meaning nothing happens)
            connectome.add_edges_from(edges)

    if out_path is not None:
        with open(out_path, 'wb') as f:
            pickle.dump(connectome, f)
    return connectome


def parse_connectivity_artificial_types_idx_based_types(data_path, artificial_types_path, full_node_list_path,
                                                        out_path):
    with open(data_path, 'rb') as f:
        connectome = pickle.load(f)
    with open(artificial_types_path, 'rb') as f:
        artificial_types = pickle.load(f)[1]
    with open(full_node_list_path, 'rb') as f:
        full_node_list = sorted(pickle.load(f))
    sorted_nodes = sorted(connectome.nodes)
    for node in sorted_nodes:
        node_idx = full_node_list.index(node)
        for art_type in range(len(artificial_types)):
            if node_idx in artificial_types[art_type]:
                node_type = art_type
                break
        connectome.nodes[node]["type"] = node_type
    with open(out_path, 'wb') as f:
        pickle.dump(connectome, f)


def parse_connectivity_noisy_birth_times_mei_zhen(data_path, birth_times_noise, out_path, num_repeats=100):
    for i in range(1, num_repeats + 1):
        noisy_birth_times = {}
        data_set = 'Dataset7.pkl'
        with open(os.path.join(data_path, data_set), 'rb') as f:
            cur_data = pickle.load(f)
        cur_data_noisy_birth_times = deepcopy(cur_data)
        if not os.path.exists(os.path.join(out_path, f'{int(100 * birth_times_noise)}%_noise')):
            os.mkdir(os.path.join(out_path, f'{int(100 * birth_times_noise)}%_noise'))
        if not os.path.exists(os.path.join(out_path, f'{int(100 * birth_times_noise)}%_noise', f'{i}')):
            os.mkdir(os.path.join(out_path, f'{int(100 * birth_times_noise)}%_noise', f'{i}'))
        for node in cur_data.nodes:
            if node not in noisy_birth_times.keys():
                true_birth_time = cur_data.nodes[node]['birth_time']
                max_noise_window_size = 2 * min(true_birth_time, ADULT_WORM_AGE - 10 - true_birth_time)
                noisy_birth_time = true_birth_time + (
                        np.random.rand() - 0.5) * max_noise_window_size * birth_times_noise
                noisy_birth_times[node] = noisy_birth_time
            cur_data_noisy_birth_times.nodes[node]['birth_time'] = noisy_birth_times[node]
        with open(os.path.join(out_path, f'{int(100 * birth_times_noise)}%_noise', f'{i}', data_set), 'wb') as f:
            pickle.dump(cur_data_noisy_birth_times, f)


def calc_average_birth_time_shift_noised_dataset(
        noised_datasets_path=os.path.join("CElegansData", "SubTypes", "noised_birth_times_connectomes"),
        data_path=os.path.join("CElegansData", "SubTypes", "connectomes", "Dataset7.pkl")):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    num_nodes = data.number_of_nodes()
    noise_levels = np.arange(0.1, 1.1, 0.1)
    average_shifts = np.zeros(len(noise_levels))
    stds_shifts = np.zeros(len(noise_levels))
    noise_level_idx = 0
    for noise_level in noise_levels:
        repeats = os.listdir(os.path.join(noised_datasets_path, f'{int(100 * noise_level)}%_noise'))
        cur_shifts = np.zeros((num_nodes, len(repeats)))
        rep_idx = 0
        for repeat in repeats:
            with open(os.path.join(noised_datasets_path, f'{int(100 * noise_level)}%_noise', repeat, "Dataset7.pkl"),
                      'rb') as f:
                noised_connectome = pickle.load(f)
            n_idx = 0
            for node in data.nodes:
                cur_shifts[n_idx, rep_idx] = np.abs(
                    data.nodes[node]["birth_time"] - noised_connectome.nodes[node]["birth_time"])
                n_idx += 1
            rep_idx += 1

        average_shifts[noise_level_idx] = cur_shifts.mean()
        stds_shifts[noise_level_idx] = cur_shifts.std()
        noise_level_idx += 1
    return average_shifts, stds_shifts


def add_noise_to_birth_times_synapses_list(synapses_lists_path, connectome_path, birth_times_noise, out_path):
    np.random.seed(int(123456789 * birth_times_noise))
    with open(connectome_path, 'rb') as f:
        connectome = pickle.load(f)
    for split in os.listdir(synapses_lists_path):
        cur_noisy_birth_times = {}
        for neuron in connectome.nodes:
            true_birth_time = connectome.nodes[neuron]['birth_time']
            max_noise_window_size = 2 * min(true_birth_time, ADULT_WORM_AGE - true_birth_time)
            noisy_birth_time = true_birth_time + (np.random.rand() - 0.5) * max_noise_window_size * birth_times_noise
            cur_noisy_birth_times[neuron] = noisy_birth_time
        with open(os.path.join(synapses_lists_path, split, 'test', 'Dataset8.pkl'), 'rb') as f:
            full_synapses_list = pickle.load(f)
        noisy_synapses_list = deepcopy(full_synapses_list)
        for i in range(len(noisy_synapses_list)):
            pre = noisy_synapses_list[i][0]
            post = noisy_synapses_list[i][1]
            noisy_synapses_list[i][-1]['birth time'] = max(cur_noisy_birth_times[pre], cur_noisy_birth_times[post])

        if not os.path.exists(os.path.join(out_path, f'{int(100 * birth_times_noise)}%_noise')):
            os.mkdir(os.path.join(out_path, f'{int(100 * birth_times_noise)}%_noise'))
        if not os.path.exists(os.path.join(out_path, f'{int(100 * birth_times_noise)}%_noise', split)):
            os.mkdir(os.path.join(out_path, f'{int(100 * birth_times_noise)}%_noise', split))
        with open(os.path.join(out_path, f'{int(100 * birth_times_noise)}%_noise', split, 'test.pkl'), 'wb') as f:
            pickle.dump(noisy_synapses_list, f)


def parse_connectivity_artificial_types_name_based_types(data_path, artificial_types_path,
                                                         out_path):
    with open(data_path, 'rb') as f:
        connectome = pickle.load(f)
    with open(artificial_types_path, 'rb') as f:
        artificial_types = pickle.load(f)[1]
    for node in connectome.nodes:
        for art_type in range(len(artificial_types)):
            if node in artificial_types[art_type]:
                node_type = art_type
                break
        connectome.nodes[node]["type"] = node_type
    with open(out_path, 'wb') as f:
        pickle.dump(connectome, f)


def parse_connectivity_mei_zhen(num_synapses_data_path, positions_path, birth_times_path, type_path, type_configuration,
                                out_dir_path):
    birth_times = pd.read_excel(birth_times_path, index_col=0)
    positions = pd.read_csv(positions_path, header=None, index_col=0)
    cell_types = pd.read_excel(type_path, index_col=0, sheet_name=None)
    for i in range(1, 9):
        cur_df = pd.read_excel(num_synapses_data_path, sheet_name=f'Dataset{i}', index_col=2, skiprows=2)
        cur_df = cur_df[1:]
        cur_df = cur_df.iloc[:, 2:]
        neurons_of_age = [n for n in cur_df.index if
                          n in birth_times.index and birth_times.loc[n][BIRTH_TIME_COL] <= FULL_DEVELOPMENTAL_AGES[
                              min(i - 1, 6)]]
        rows_to_drop = [n for n in cur_df.index if n not in neurons_of_age]
        cols_to_drop = [n for n in cur_df if n not in neurons_of_age]
        cur_df.drop(rows_to_drop, axis=0, inplace=True)
        cur_df.drop(cols_to_drop, axis=1, inplace=True)
        cur_connectome = nx.empty_graph(0, create_using=nx.DiGraph)
        for n in neurons_of_age:
            if CElegansNeuronsAdder.SINGLE_TYPE == type_configuration:
                neuron_type = None
            else:
                for sheet in cell_types.keys():
                    if n in cell_types[sheet].index:
                        neuron_type_xls = cell_types[sheet].loc[n, CELL_TYPE_COL]
                        if neuron_type_xls in SENSORY_XLS:
                            neuron_type = SENSORY
                        elif neuron_type_xls in MOTOR_XLS:
                            neuron_type = MOTOR
                        elif neuron_type_xls in INTER_XLS:
                            neuron_type = INTER
                        if CElegansNeuronsAdder.COARSE_TYPES == type_configuration:
                            break
                        elif CElegansNeuronsAdder.SUB_TYPES == type_configuration:
                            neurons_subtype_xls = cell_types[sheet].loc[n, CELL_SUBTYPE_COL]
                            if neurons_subtype_xls in CELL_TYPE_MAPPING.keys():
                                neuron_type = CELL_TYPE_MAPPING[neurons_subtype_xls]
                        break
            cur_connectome.add_node(n,
                                    coords=np.array([positions.loc[n][1], positions.loc[n][2],
                                                     positions.loc[n][3]]) * 1 / WORM_LENGTH_NORMALIZATION,
                                    birth_time=birth_times.loc[n, BIRTH_TIME_COL],
                                    type=neuron_type)
        for pre in cur_df.index:
            for post in cur_df.index:
                if pre == post:
                    continue
                if cur_df.loc[post, pre] > 0:
                    length = np.linalg.norm(cur_connectome.nodes[post]['coords'] - cur_connectome.nodes[pre]['coords'])
                    birth_time = max(cur_connectome.nodes[post]['birth_time'], cur_connectome.nodes[pre]['birth_time'])
                    edge = (pre, post, {'length': length, 'birth time': birth_time})
                    cur_connectome.add_edges_from([edge])
        out_path = os.path.join(out_dir_path, f'Dataset{i}' + ".pkl")
        with open(out_path, 'wb') as f:
            pickle.dump(cur_connectome, f)


def validate_neuronal_coordinates_in_parsed_data(neuronal_positions_3d_path, parsed_data_path):
    with open(parsed_data_path, 'rb') as f:
        parsed_data = pickle.load(f)
    positions_data = pd.read_csv(neuronal_positions_3d_path, header=None)
    for neuron in parsed_data.nodes:
        cur_parsed_coordinates = parsed_data.nodes[neuron]['coords']
        cur_neuron_row = positions_data.loc[positions_data[0] == neuron]
        for i in range(3):
            cur_coordinate = cur_neuron_row.iloc[0, i + 1] / WORM_LENGTH_NORMALIZATION
            if cur_coordinate != cur_parsed_coordinates[i]:
                print(neuron)


def validate_synaptic_lengths_in_parsed_data(neuronal_positions_3d_path, parsed_data_path):
    with open(parsed_data_path, 'rb') as f:
        parsed_data = pickle.load(f)
    positions_data = pd.read_csv(neuronal_positions_3d_path, header=None)
    for synapse in parsed_data.edges:
        pre_neuron = synapse[0]
        post_neuron = synapse[1]
        cur_parsed_len = parsed_data.get_edge_data(pre_neuron, post_neuron)['length']
        cur_pre_neuron_row = positions_data.loc[positions_data[0] == pre_neuron]
        cur_post_neuron_row = positions_data.loc[positions_data[0] == post_neuron]
        true_length = 0
        for i in range(3):
            cur_coordinate_pre = cur_pre_neuron_row.iloc[0, i + 1] / WORM_LENGTH_NORMALIZATION
            cur_coordinate_post = cur_post_neuron_row.iloc[0, i + 1] / WORM_LENGTH_NORMALIZATION
            true_length += (cur_coordinate_pre - cur_coordinate_post) ** 2
        true_length = np.sqrt(true_length)
        if true_length != cur_parsed_len:
            print(synapse)


def create_sub_connectome_with_matching_nodes(data_set_full_path, data_set_sub_path, out_path):
    with open(data_set_full_path, 'rb') as f:
        worm_atlas_data_set = pickle.load(f)
    with open(data_set_sub_path, 'rb') as f:
        mei_jhen_data_set = pickle.load(f)
    mei_jhen_neurons = list(mei_jhen_data_set.nodes)
    worm_atlas_neurons = list(worm_atlas_data_set.nodes)
    for neuron in worm_atlas_neurons:
        if neuron not in mei_jhen_neurons:
            worm_atlas_data_set.remove_node(neuron)
    with open(out_path, 'wb') as f:
        pickle.dump(worm_atlas_data_set, f)


def save_synaptic_subtypes():
    counter = 0
    synaptic_subtypes = {}
    for pre_type in NEURONAL_SUBTYPES:
        for post_type in NEURONAL_SUBTYPES:
            cur_type = (pre_type, post_type)
            synaptic_subtypes[cur_type] = counter
            counter += 1
    with open("SavedOutputs\\CElegansModel\\synaptic_subtypes_dict.pkl", 'wb') as f:
        pickle.dump(synaptic_subtypes, f)


def find_neuronal_type(neuron, type_configuration, type_path="CElegansData\\CellTypes.xlsx"):
    cell_types = pd.read_excel(type_path, index_col=0, sheet_name=None)
    neuron_type = None
    for sheet in cell_types.keys():
        if neuron in cell_types[sheet].index:
            neuron_type_xls = cell_types[sheet].loc[
                neuron, CELL_TYPE_COL]
            if neuron_type_xls in SENSORY_XLS:
                neuron_type = SENSORY
            elif neuron_type_xls in MOTOR_XLS:
                neuron_type = MOTOR
            elif neuron_type_xls in INTER_XLS:
                neuron_type = INTER
            if CElegansNeuronsAdder.COARSE_TYPES == type_configuration:
                break
            elif CElegansNeuronsAdder.SUB_TYPES == type_configuration:
                neurons_subtype_xls = cell_types[sheet].loc[neuron, CELL_SUBTYPE_COL]
                if neurons_subtype_xls in CELL_TYPE_MAPPING.keys():
                    neuron_type = CELL_TYPE_MAPPING[neurons_subtype_xls]
            break
    return neuron_type


def find_neuronal_artificial_type(neuron_alphabet_idx, types_path):
    with open(types_path, 'rb') as f:
        neuronal_types = pickle.load(f)[1]
    type_idx = 0
    for n_type in neuronal_types:
        if neuron_alphabet_idx in n_type:
            return type_idx
        type_idx += 1


def get_all_neuronal_artificial_types(num_neurons, types_path):
    neuronal_types = {}
    for neuronal_alphabetic_idx in range(num_neurons):
        neuronal_types[neuronal_alphabetic_idx] = find_neuronal_artificial_type(neuronal_alphabetic_idx, types_path)
    return neuronal_types


def find_neuronal_coordinates(neuron):
    dummy = CElegansNeuronsAdder(CElegansNeuronsAdder.SUB_TYPES)
    neurons_df = dummy.get_neurons_df()
    return neurons_df[CElegansNeuronsAdder.POSITION_STR][neuron]


def get_coordinates_of_all_neurons():
    dummy = CElegansNeuronsAdder(CElegansNeuronsAdder.SUB_TYPES)
    neurons_df = dummy.get_neurons_df()
    return neurons_df[CElegansNeuronsAdder.POSITION_STR]


def sort_neurons_by_biological_types(neurons_list_path):
    cook_types_list, _ = create_cook_types_list(neurons_list_path)
    neurons_names_by_type = []
    for n_type in cook_types_list:
        neurons_names_by_type += sorted(list(n_type))
    with open(neurons_list_path, 'rb') as f:
        alphabetic_neuronal_names = sorted(pickle.load(f))
    neurons_idx_by_type = np.zeros((len(alphabetic_neuronal_names), 1)).astype(int)
    for j in range(len(alphabetic_neuronal_names)):
        neurons_idx_by_type[j] = alphabetic_neuronal_names.index(neurons_names_by_type[j])
    return neurons_idx_by_type


def sort_neurons_by_artificial_types(neuronal_types_path, neurons_list_for_type_aggregation_path,
                                     actual_neurons_in_dataset_list):
    neuronal_types_by_names = convert_indices_to_names_in_artificial_types(neuronal_types_path,
                                                                           neurons_list_for_type_aggregation_path)
    neuronal_names_ordered_by_type = []
    for neuronal_type in neuronal_types_by_names:
        neuronal_names_ordered_by_type += sorted(list(neuronal_type))

    alphabetic_neuronal_names = actual_neurons_in_dataset_list

    with open(neurons_list_for_type_aggregation_path, 'rb') as f:
        neurons_list_for_type_aggregation = pickle.load(f)

    for n in neurons_list_for_type_aggregation:
        if n not in alphabetic_neuronal_names:
            neuronal_names_ordered_by_type.remove(n)

    neurons_idx_by_type = np.zeros((len(alphabetic_neuronal_names), 1)).astype(int)
    for j in range(len(alphabetic_neuronal_names)):
        neurons_idx_by_type[j] = alphabetic_neuronal_names.index(neuronal_names_ordered_by_type[j])
    return neurons_idx_by_type


def convert_indices_to_names_in_artificial_types(art_type_path, neurons_list_path):
    with open(art_type_path, 'rb') as f:
        art_types = pickle.load(f)[1]
    with open(neurons_list_path, 'rb') as f:
        neurons_list = sorted(pickle.load(f))
    art_types_by_names = []
    for art_type in art_types:
        art_type_names = set([neurons_list[idx] for idx in art_type])
        art_types_by_names.append(art_type_names)
    return art_types_by_names


def convert_names_to_indices_neuronal_types(types_by_names, neurons_list_path):
    with open(neurons_list_path, 'rb') as f:
        neurons_list = pickle.load(f)
    types_by_indices = []
    for named_type in types_by_names:
        type_by_indices = set([neurons_list.index(name) for name in named_type if name in neurons_list])
        types_by_indices.append(type_by_indices)
    return types_by_indices


def convert_indices_of_types_between_neuronal_lists(types_by_indices_1_path, neuronal_list_1_path,
                                                    neuronal_list_2_path):
    with open(types_by_indices_1_path, 'rb') as f:
        types_by_indices_1 = pickle.load(f)[1]
    with open(neuronal_list_1_path, 'rb') as f:
        neuronal_list_1 = sorted(pickle.load(f))
    with open(neuronal_list_2_path, 'rb') as f:
        neuronal_list_2 = sorted(pickle.load(f))
    types_by_indices_2 = []
    for n_type in types_by_indices_1:
        cur_converted_type = set([])
        for idx in n_type:
            neuron = neuronal_list_1[idx]
            if neuron in neuronal_list_2:
                new_idx = neuronal_list_2.index(neuron)
                cur_converted_type = cur_converted_type.union({new_idx})
        types_by_indices_2.append(cur_converted_type)
    return types_by_indices_2


def create_cook_types_list(neurons_list_path):
    with open('CElegansData\\neuronal_types_dict.pkl', 'rb') as f:
        neuronal_types_dict = pickle.load(f)
    with open(neurons_list_path, 'rb') as f:
        neurons_list = pickle.load(f)

    list_of_neuronal_groups = []
    list_of_type_names = []
    for neuron in neuronal_types_dict.keys():
        if neuron not in neurons_list:
            continue
        cur_type = neuronal_types_dict[neuron][CElegansNeuronsAdder.SUB_TYPES]
        if cur_type not in list_of_type_names:
            list_of_type_names.append(cur_type)
            list_of_neuronal_groups.append(set([neuron]))
        else:
            type_idx = list_of_type_names.index(cur_type)
            list_of_neuronal_groups[type_idx] = list_of_neuronal_groups[type_idx].union(set([neuron]))

    return list_of_neuronal_groups, list_of_type_names


def find_neurons_subset_at_age(reference_age, neurons_subset_path):
    dummy = CElegansNeuronsAdder(CElegansNeuronsAdder.SUB_TYPES, neurons_sub_set_path=neurons_subset_path)
    neurons_df = dummy.get_neurons_df()
    neurons_df = neurons_df.loc[neurons_df[CElegansNeuronsAdder.BIRTH_TIME_STR] < reference_age]
    return sorted(neurons_df.index)


def split_all_possible_synapses_to_reciprocal_train_and_test(all_data_sets_path, out_path):
    # np.random.seed(123456789)
    with open(os.path.join(all_data_sets_path, "Dataset7.pkl"), 'rb') as f:
        cur_data_set = pickle.load(f)
    sorted_neurons = sorted(cur_data_set.nodes)
    num_neurons = len(sorted_neurons)
    upper_triangle_synapses = []
    lower_triangle_synapses = []
    for pre_idx in range(num_neurons - 1):
        pre = sorted_neurons[pre_idx]
        for post_idx in range(pre_idx + 1, num_neurons):
            post = sorted_neurons[post_idx]
            birth_time = max(cur_data_set.nodes[pre]['birth_time'], cur_data_set.nodes[post]['birth_time'])
            length = np.sqrt(np.sum((cur_data_set.nodes[pre]['coords'] - cur_data_set.nodes[post]['coords']) ** 2))
            upper_type = (cur_data_set.nodes[pre]['type'], cur_data_set.nodes[post]['type'])
            lower_type = (cur_data_set.nodes[post]['type'], cur_data_set.nodes[pre]['type'])
            upper_triangle_synapses.append(
                (pre, post, {'birth time': birth_time, 'length': length, 'type': upper_type}))
            lower_triangle_synapses.append(
                (post, pre, {'birth time': birth_time, 'length': length, 'type': lower_type}))

    num_upper_synapses = len(upper_triangle_synapses)

    train_indices = np.random.choice(num_upper_synapses, size=num_upper_synapses // 2,
                                     replace=False).astype(int)
    train_indices = np.sort(train_indices)
    test_indices = np.array([i for i in range(num_upper_synapses) if i not in train_indices])
    train_subset = [upper_triangle_synapses[idx] for idx in train_indices] + [lower_triangle_synapses[idx] for idx in
                                                                              train_indices]
    test_subset = [upper_triangle_synapses[idx] for idx in test_indices] + [lower_triangle_synapses[idx] for idx in
                                                                            test_indices]
    for data_set in os.listdir(all_data_sets_path):
        with open(os.path.join(all_data_sets_path, data_set), 'rb') as f:
            cur_data_set = pickle.load(f)
        cur_neurons = cur_data_set.nodes
        cur_train_subset = []
        cur_test_subset = []
        for synapse in train_subset:
            syn_cpy = deepcopy(synapse)
            if synapse[0] in cur_neurons and synapse[1] in cur_neurons:
                syn_cpy[-1]['exists'] = 1 if (synapse[0], synapse[1]) in cur_data_set.edges else 0
                cur_train_subset.append(syn_cpy)
        for synapse in test_subset:
            syn_cpy = deepcopy(synapse)
            if synapse[0] in cur_neurons and synapse[1] in cur_neurons:
                syn_cpy[-1]['exists'] = 1 if (synapse[0], synapse[1]) in cur_data_set.edges else 0
                cur_test_subset.append(syn_cpy)

        train_path = os.path.join(out_path, "train")
        if not os.path.exists(train_path):
            os.mkdir(train_path)
        test_path = os.path.join(out_path, "test")
        if not os.path.exists(test_path):
            os.mkdir(test_path)
        with open(os.path.join(train_path, f"{data_set}"), 'wb') as f:
            pickle.dump(cur_train_subset, f)
        with open(os.path.join(test_path, f"{data_set}"), 'wb') as f:
            pickle.dump(cur_test_subset, f)


def validate_types_in_connectome(connectome_path, types_path, full_neurons_list_path):
    with open(connectome_path, 'rb') as f:
        connectome = pickle.load(f)
    with open(types_path, 'rb') as f:
        types = pickle.load(f)[1]
    with open(full_neurons_list_path, 'rb') as f:
        full_neurons_list = sorted(pickle.load(f))
    cur_neurons = sorted(connectome.nodes)
    for neuron in cur_neurons:
        type_in_connectome = connectome.nodes[neuron]['type']
        neuron_idx_in_list = full_neurons_list.index(neuron)
        for type_idx in range(len(types)):
            cur_type = types[type_idx]
            if neuron_idx_in_list in cur_type:
                type_in_types_file = type_idx
                break
        if type_in_types_file != type_in_connectome:
            print(neuron + " does not match")


def construct_exists_array_from_syn_list(data_path):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    exists = np.zeros(len(data))
    for i in range(len(data)):
        exists[i] = data[i][-1]['exists']
    return exists
