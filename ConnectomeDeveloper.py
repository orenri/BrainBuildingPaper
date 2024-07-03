import numpy as np
import networkx as nx
import copy
from c_elegans_constants import C_ELEGANS_LENGTHS


# A class that simulates the dynamics of a connectome.
class ConnectomeDeveloper:
    DEFAULT_INITIAL_STATE = nx.from_numpy_array(np.zeros((2, 2)), create_using=nx.DiGraph)
    # The network is embedded in a physical space - the cube [0,1)^d, where d is the number of dimensions. The neuronal
    # attribute "coords" refers to the coordinates of neurons in this space.
    DEFAULT_INITIAL_NODE_ATTRIBUTES = {3: {0: {"coords": np.array([[0.25], [0.25], [0.25]])},
                                           1: {"coords": np.array([[0.75], [0.75], [0.75]])}},
                                       2: {0: {"coords": np.array([[0.25], [0.25]])},
                                           1: {"coords": np.array([[0.75], [0.75]])}}}
    DEFAULT_INITIAL_EDGE_ATTRIBUTES = {}

    def __init__(self, operators, time_step=1, initial_state=DEFAULT_INITIAL_STATE, dimension=3,
                 initial_node_attributes=None,
                 initial_edge_attributes=DEFAULT_INITIAL_EDGE_ATTRIBUTES,
                 sizes_by_age=C_ELEGANS_LENGTHS):
        self.operators_ = operators
        self.t_ = 0
        self.dt_ = time_step  # Arbitrary units
        # The connectome is a networkx instance
        self.connectome_ = initial_state
        if initial_node_attributes is None:
            nx.set_node_attributes(self.connectome_, self.DEFAULT_INITIAL_NODE_ATTRIBUTES[dimension])
        else:
            nx.set_node_attributes(self.connectome_, initial_node_attributes)
        nx.set_edge_attributes(self.connectome_, initial_edge_attributes)

        self.previous_connectome_ = self.connectome_.copy()
        self.num_neurons_ = self.connectome_.number_of_nodes()
        self.num_synapses_ = self.connectome_.number_of_edges()
        self.dimension_ = dimension
        # Normalized size of the animal in the beginning of the development, allows isotropic growth throughout it.
        self.growth_factor_ = 1
        # Measured lengths of the animal in different ages.
        self.sizes_by_age_ = sizes_by_age

    def update_state(self):
        for op in self.operators_:
            # Each operator must refer to the state of the connectome from the beginning of the iteration (before other
            # operators have operated), for avoiding importance of the order of the operators list.
            new_connectome = op.operate(self)
            self.connectome_ = new_connectome
        self.num_neurons_ = self.connectome_.number_of_nodes()
        self.num_synapses_ = self.connectome_.number_of_edges()
        self.previous_connectome_ = self.connectome_.copy()

    def simulate(self, duration):
        num_steps = int(duration / self.dt_)
        for i in range(num_steps):
            # The time and the norm size must be updated before operation of a proper time being used when operating
            self.t_ += self.dt_
            self.update_growth_factor_piecewise_linear()
            self.update_state()

    def update_growth_factor_piecewise_linear(self):
        ages_of_measured_size = sorted(self.sizes_by_age_.keys())
        cur_growth_stage = 1
        for age in ages_of_measured_size[1:]:
            if self.t_ <= age:
                break
            cur_growth_stage += 1
        previous_measured_age = ages_of_measured_size[cur_growth_stage - 1]
        next_measured_age = ages_of_measured_size[cur_growth_stage]
        previous_size = self.sizes_by_age_[previous_measured_age]
        next_size = self.sizes_by_age_[next_measured_age]

        cur_size = previous_size + (self.t_ - previous_measured_age) * (next_size - previous_size) / (
                next_measured_age - previous_measured_age)
        self.growth_factor_ = cur_size / self.sizes_by_age_[ages_of_measured_size[0]]

    def get_connectome(self):
        return copy.deepcopy(self.connectome_)

    def get_num_neurons(self):
        return self.num_neurons_

    def get_num_synapses(self):
        return self.num_synapses_

    def get_previous_connectome(self):
        return copy.deepcopy(self.previous_connectome_)

    def get_dimension(self):
        return self.dimension_

    def get_time(self):
        return self.t_

    def get_growth_factor(self):
        return self.growth_factor_
