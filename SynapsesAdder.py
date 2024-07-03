from Operator import Operator
import numpy as np
import networkx as nx


# A class that adds edges (synapses) to the connectome with a given probability. Each possible synapse in the given
# connectome has a probability to be facilitated that is dependent on the baseline (S+) and the length of the synapse.
# The decay of the probability with length is exponential.
# Factors for the probability to have certain types of connections can be configured. motor2sensory and inter2inter
# connections are the ones that have the largest deviation from uniform distribution aross connection types in C.
# elegans.
class SynapsesAdder(Operator):
    def __init__(self, addition_probabilities_across_development, developmental_ages, length_decay, reciprocal_factor):
        # addition_probabilities_across_development is a dictionary of dictionaries. The outer dictionary keys are
        # developmental stages (integers), which are also the keys for developmental_ages, of which values are the ages
        # of stage transition. In the nested dictionaries in addition_probabilities_across_development the keys are
        # possible synaptic types and values are S+s for each type.
        self.addition_probabilities_across_development_ = addition_probabilities_across_development
        self.cur_developmental_stage_ = sorted(developmental_ages.keys())[0]
        self.p_ = self.addition_probabilities_across_development_[self.cur_developmental_stage_]
        self.length_decay_ = length_decay
        self.reciprocal_factor_ = reciprocal_factor
        self.developmental_ages_ = developmental_ages
        self.total_added_synapses_ = []

    def operate(self, connectome_dev):
        connectome = connectome_dev.get_connectome()
        prev_connectome = connectome_dev.get_previous_connectome()
        age = connectome_dev.get_time()
        # Update the developmental stage and the addition probabilities if time has come and we are not in the last
        # stage yet.
        if age >= self.developmental_ages_[self.cur_developmental_stage_] and self.cur_developmental_stage_ < len(
                self.developmental_ages_.keys()) - 1:
            self.cur_developmental_stage_ += 1
            self.p_ = self.addition_probabilities_across_development_[self.cur_developmental_stage_]
        # Get the current growth factor of the embedding space of the network (the ratio between the initial size of the
        # animal and its current size assuming isotropic growth)
        growth_factor = connectome_dev.get_growth_factor()
        # Correct the decay constant according to the current size of the animal (as the distances are normalized to its
        #  size).
        cur_length_decay = self.length_decay_ * growth_factor
        possible_synapses = list(nx.non_edges(prev_connectome))
        probs = np.zeros(len(possible_synapses))
        lengths = np.zeros(len(possible_synapses))
        for i in range(len(possible_synapses)):
            lengths[i] = np.sqrt(np.sum((prev_connectome.nodes[possible_synapses[i][0]]["coords"] -
                                         prev_connectome.nodes[possible_synapses[i][1]]["coords"]) ** 2))
            synaptic_type = (prev_connectome.nodes[possible_synapses[i][0]]["type"],
                             prev_connectome.nodes[possible_synapses[i][1]]["type"])
            probs[i] = self.p_[synaptic_type] * np.exp(-lengths[i] * cur_length_decay)
            cur_reciprocal_synapse = (possible_synapses[i][1], possible_synapses[i][0])
            if cur_reciprocal_synapse in prev_connectome.edges:
                probs[i] = min(1, probs[i] * self.reciprocal_factor_)
        # sample a binary vector for indicating which synapses to facilitate (each entry is 1 with probability p,
        # indicating an index in the array of indices of possible synapses chosen to be added).
        added_synapses_indices = np.nonzero(np.random.binomial(1, probs, len(possible_synapses)))[0]
        added_synapses = []
        for index in added_synapses_indices:
            cur_syn = possible_synapses[index]
            cur_length = lengths[index]
            added_synapses.append((cur_syn[0], cur_syn[1], {"length": cur_length}))
        connectome.add_edges_from(added_synapses)
        self.total_added_synapses_.append(len(added_synapses))

        return connectome

    def get_total_added_synapses(self):
        return self.total_added_synapses_
