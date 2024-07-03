from Operator import Operator
import numpy as np
import os
import pandas as pd
import pickle
from c_elegans_constants import WORM_LENGTH_NORMALIZATION


# A class that adds nodes (neurons) to the connectome according to the birth times of c. elegans neurons.
class CElegansNeuronsAdder(Operator):
    DEFAULT_DATA_PATH = "CElegansData"
    DEFAULT_BIRTH_TIMES_FILE_NAME = "time_of_birth.xls"
    DEFAULT_POSITIONS_FILE_NAME_1D = "NeuronPosition.xls"
    DEFAULT_POSITIONS_FILE_NAME_3D = "NeuronPosition3D.csv"
    DEFAULT_TYPES_FILE_NAME = "CellTypes.xlsx"

    # Columns and strings in the xls files
    POSITION_COL = "Soma Position"
    BIRTH_TIME_COL = "Birth time (min.)"
    CELL_TYPE_COL = "cell type"
    CELL_SUBTYPE_COL = "cell category"

    # Neuronal types in cell types excel
    SENSORY_XLS = ["sensory neuron", "sensory"]
    MOTOR_XLS = ["motorneuron"]
    INTER_XLS = ["interneuron"]

    POSITION_STR = "Position"
    BIRTH_TIME_STR = "BirthTime"  # in minutes
    TYPE_STR = "Type"

    # Neuronal subtypes in cell types excel
    SENSORY1_XLS = "SN1"
    SENSORY2_XLS = "SN2"
    SENSORY3_XLS = "SN3"
    SENSORY4_XLS = "SN4"
    SENSORY5_XLS = "SN5"
    SENSORY6_XLS = "SN6"
    MOTOR1_XLS = "head motor neuron"
    MOTOR2_XLS = "sublateral motor neuron"
    MOTOR3_XLS = "ventral cord motor neuron"
    MOTOR4_XLS = "sex-specific neuron"
    INTER1_XLS = "layer 1 interneuron"
    INTER2_XLS = "layer 2 interneuron"
    INTER3_XLS = "layer 3 interneuron"
    INTER4_XLS = "category 4 interneuron"
    INTER5_XLS = "linker to pharynx"

    # Neuronal types in connectome
    SENSORY = "S"
    SENSORY1 = "S1"
    SENSORY2 = "S2"
    SENSORY3 = "S3"
    SENSORY4 = "S4"
    SENSORY5 = "S5"
    SENSORY6 = "S6"
    MOTOR = "M"
    MOTOR1 = "M1"
    MOTOR2 = "M2"
    MOTOR3 = "M3"
    MOTOR4 = "M4"
    INTER = "I"
    INTER1 = "I1"
    INTER2 = "I2"
    INTER3 = "I3"
    INTER4 = "I4"
    INTER5 = "I5"

    # Mapping between excel file subtypes and code subtypes
    CELL_TYPE_MAPPING = {SENSORY1_XLS: SENSORY1, SENSORY2_XLS: SENSORY2, SENSORY3_XLS: SENSORY3, SENSORY4_XLS: SENSORY4,
                         SENSORY5_XLS: SENSORY5, SENSORY6_XLS: SENSORY6, MOTOR1_XLS: MOTOR1, MOTOR2_XLS: MOTOR2,
                         MOTOR3_XLS: MOTOR3, MOTOR4_XLS: MOTOR4, INTER1_XLS: INTER1, INTER2_XLS: INTER2,
                         INTER3_XLS: INTER3,
                         INTER4_XLS: INTER4, INTER5_XLS: INTER5}

    # Different configurations of  neuronal types for anlysis
    SINGLE_TYPE = 0
    COARSE_TYPES = 1
    SUB_TYPES = 2
    ARTIFICIAL_TYPES = 3

    def __init__(self, type_configuration, c_elegans_data_path=DEFAULT_DATA_PATH,
                 birth_times_file=DEFAULT_BIRTH_TIMES_FILE_NAME,
                 positions_file_name=DEFAULT_POSITIONS_FILE_NAME_3D,
                 types_file_name=DEFAULT_TYPES_FILE_NAME, neurons_sub_set_path=None):
        # Build a data frame containing all the C. elegans neurons, their birth times (in minutes), their positions
        # (normalized) and their types
        if 'xls' in birth_times_file:
            birth_times = pd.read_excel(os.path.join(c_elegans_data_path, birth_times_file), index_col=0)
        elif 'csv' in birth_times_file:
            birth_times = pd.read_csv(os.path.join(c_elegans_data_path, birth_times_file), index_col=0)
        self.neurons_df_ = pd.DataFrame()
        self.neurons_df_.insert(0, CElegansNeuronsAdder.BIRTH_TIME_STR,
                                birth_times[CElegansNeuronsAdder.BIRTH_TIME_COL])

        if positions_file_name == CElegansNeuronsAdder.DEFAULT_POSITIONS_FILE_NAME_3D:
            positions = pd.read_csv(os.path.join(c_elegans_data_path, positions_file_name), header=None)
            positions.drop(
                [idx for idx in range(len(positions.index)) if positions.iloc[idx, 0] not in birth_times.index.values],
                inplace=True)
            coords_list = [1 / WORM_LENGTH_NORMALIZATION * np.array(
                [positions.iloc[i, 1], positions.iloc[i, 2], positions.iloc[i, 3]]) for i in
                           range(len(positions.index))]
            self.neurons_df_.insert(1, CElegansNeuronsAdder.POSITION_STR, coords_list)
        else:
            positions = pd.read_excel(os.path.join(c_elegans_data_path, positions_file_name), index_col=0)
            self.neurons_df_.insert(1, CElegansNeuronsAdder.POSITION_STR, positions[CElegansNeuronsAdder.POSITION_COL])

        # Remove all neurons that aren't in the given subset (if given)
        # This must happen before the types are assigned as artificial types use nodes indices (after sorting in an
        # alphabetical order) as neuronal names, so if we use indices of a subset but assign for all types will get
        # messed up.
        if neurons_sub_set_path is not None:
            with open(neurons_sub_set_path, 'rb') as f:
                neurons_sub_set = sorted(pickle.load(f))
            self.neurons_df_.drop([neuron for neuron in self.neurons_df_.index if neuron not in neurons_sub_set],
                                  inplace=True)

        if CElegansNeuronsAdder.SINGLE_TYPE == type_configuration:
            neuronal_types_list = [None] * len(self.neurons_df_.index)
        elif CElegansNeuronsAdder.ARTIFICIAL_TYPES == type_configuration:
            with open(os.path.join(c_elegans_data_path, types_file_name), 'rb') as f:
                artificial_types = pickle.load(f)[1]
            neuronal_types_list = []
            for node_idx in range(len(self.neurons_df_.index)):
                for art_type in range(len(artificial_types)):
                    if node_idx in artificial_types[art_type]:
                        node_type = art_type
                        break
                neuronal_types_list.append(node_type)
        else:
            types = pd.read_excel(os.path.join(c_elegans_data_path, types_file_name), index_col=0, sheet_name=None)
            neuronal_types_list = []
            for neuron in self.neurons_df_.index:
                for sheet in types.keys():
                    if neuron in types[sheet].index:
                        cur_type = types[sheet].loc[neuron, CElegansNeuronsAdder.CELL_TYPE_COL]
                        cur_subtype = types[sheet].loc[neuron, CElegansNeuronsAdder.CELL_SUBTYPE_COL]
                        if cur_type in CElegansNeuronsAdder.SENSORY_XLS:
                            if CElegansNeuronsAdder.SUB_TYPES == type_configuration and cur_subtype in CElegansNeuronsAdder.CELL_TYPE_MAPPING.keys():
                                neuronal_types_list.append(CElegansNeuronsAdder.CELL_TYPE_MAPPING[cur_subtype])
                            else:
                                neuronal_types_list.append(CElegansNeuronsAdder.SENSORY)
                        elif cur_type in CElegansNeuronsAdder.MOTOR_XLS:
                            if CElegansNeuronsAdder.SUB_TYPES == type_configuration and cur_subtype in CElegansNeuronsAdder.CELL_TYPE_MAPPING.keys():
                                neuronal_types_list.append(CElegansNeuronsAdder.CELL_TYPE_MAPPING[cur_subtype])
                            else:
                                neuronal_types_list.append(CElegansNeuronsAdder.MOTOR)
                        elif cur_type in CElegansNeuronsAdder.INTER_XLS:
                            if CElegansNeuronsAdder.SUB_TYPES == type_configuration and cur_subtype in CElegansNeuronsAdder.CELL_TYPE_MAPPING.keys():
                                neuronal_types_list.append(CElegansNeuronsAdder.CELL_TYPE_MAPPING[cur_subtype])
                            else:
                                neuronal_types_list.append(CElegansNeuronsAdder.INTER)
                        break
        self.neurons_df_.insert(2, CElegansNeuronsAdder.TYPE_STR, neuronal_types_list)

        # Sort neurons (rows of the data frame) according to their birth times
        self.neurons_df_.sort_values(by=[CElegansNeuronsAdder.BIRTH_TIME_STR], inplace=True)

        # Set the index of the next neuron to add to be 0
        self.next_to_add_ = 0

    def operate(self, connectome_dev):
        connectome = connectome_dev.get_connectome()
        time = connectome_dev.get_time()
        next_birth_time = self.neurons_df_[CElegansNeuronsAdder.BIRTH_TIME_STR][self.next_to_add_]

        # Check whether it is time to add the next neuron
        if time < next_birth_time:
            pass
        else:
            # Create a list of all nodes to be added (that haven't been added yet and with birth times smaller than the
            # current time)
            nodes_to_add = []
            while next_birth_time <= time:
                nodes_to_add.append((self.neurons_df_.index[self.next_to_add_], {
                    "coords": self.neurons_df_[CElegansNeuronsAdder.POSITION_STR][self.next_to_add_],
                    "type": self.neurons_df_[CElegansNeuronsAdder.TYPE_STR][self.next_to_add_],
                    "birth_time": self.neurons_df_[CElegansNeuronsAdder.BIRTH_TIME_STR][self.next_to_add_]}))
                # Update the index of the next neuron to be added until it reaches the last neuron (then keep it like
                # that for preventing an error in next iteration).
                self.next_to_add_ = self.next_to_add_ + 1
                if self.next_to_add_ > len(self.neurons_df_.index) - 1:
                    self.next_to_add_ = len(self.neurons_df_.index) - 1
                    break
                next_birth_time = self.neurons_df_[CElegansNeuronsAdder.BIRTH_TIME_STR][self.next_to_add_]
            connectome.add_nodes_from(nodes_to_add)

        return connectome

    def get_neurons_df(self):
        return self.neurons_df_
