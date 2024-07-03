from Operator import Operator
import numpy as np


# A class that removes edges (synapses) from the connectome with a given probability. Each existing synapse in the given
# connectome has a constant probability to be removed.
class SynapsesRemover(Operator):
    def __init__(self, removal_probability):
        self.p_ = removal_probability

    def operate(self, connectome_dev):
        connectome = connectome_dev.get_connectome()
        prev_connectome = connectome_dev.get_previous_connectome()
        num_synapses = connectome_dev.get_num_synapses()
        # sample a binary vector for indicating which synapses to remove (each entry is 1 with probability p,
        # indicating an index in the array of indices of existing synapses chosen to be removed).
        removed_synapses_indices = np.nonzero(np.random.binomial(1, self.p_, num_synapses))[0]
        removed_synapses = [list(prev_connectome.edges)[index] for index in removed_synapses_indices]
        connectome.remove_edges_from(removed_synapses)

        return connectome
