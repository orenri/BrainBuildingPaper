import numpy as np
import networkx as nx
from scipy.special import comb
from poisson_binomial import PoissonBinomial

NORM_DEGREE_DIST_RES = 0.05
NUM_TRIAD_TYPES = 16
SORTED_TRIAD_TYPES = sorted(list(nx.algorithms.triads.triadic_census(
    nx.convert_matrix.from_numpy_array(np.zeros((2, 2)), create_using=nx.DiGraph)).keys()))


def degree_distribution(graph):
    num_neurons = graph.number_of_nodes()
    nodes = list(graph.nodes)
    out_degrees = np.array([graph.out_degree(node) for node in nodes])
    in_degrees = np.array([graph.in_degree(node) for node in nodes])
    out_deg_dist = 1 / num_neurons * np.histogram(out_degrees, bins=np.arange(num_neurons + 1))[0]
    in_deg_dist = 1 / num_neurons * np.histogram(in_degrees, bins=np.arange(num_neurons + 1))[0]
    return in_deg_dist, out_deg_dist


def normalized_degree_distribution(graph):
    num_neurons = graph.number_of_nodes()
    nodes = list(graph.nodes)
    norm_out_degrees = np.array([graph.out_degree(node) for node in nodes]) / num_neurons
    norm_in_degrees = np.array([graph.in_degree(node) for node in nodes]) / num_neurons
    norm_out_deg_dist = 1 / num_neurons * np.histogram(norm_out_degrees, bins=np.arange(0, 1 + NORM_DEGREE_DIST_RES,
                                                                                        NORM_DEGREE_DIST_RES))[0]
    norm_in_deg_dist = 1 / num_neurons * \
                       np.histogram(norm_in_degrees, bins=np.arange(0, 1 + NORM_DEGREE_DIST_RES, NORM_DEGREE_DIST_RES))[
                           0]
    return norm_in_deg_dist, norm_out_deg_dist


def average_degree_distribution(average_adj_mat):
    num_neurons = average_adj_mat.shape[0]
    out_degrees_distributions = []
    for row in range(num_neurons):
        out_degrees_distributions.append(PoissonBinomial(average_adj_mat[row, :]))
    in_degrees_distributions = []
    for col in range(num_neurons):
        in_degrees_distributions.append(PoissonBinomial(average_adj_mat[:, col]))
    average_out_deg_hist = np.zeros(num_neurons)
    out_deg_hist_std = np.zeros(num_neurons)
    average_in_deg_hist = np.zeros(num_neurons)
    in_deg_hist_std = np.zeros(num_neurons)
    for deg in range(num_neurons):
        for neuron in range(num_neurons):
            p_neuron_has_out_degree_deg = np.clip(out_degrees_distributions[neuron].pmf[deg], a_min=0, a_max=1)
            average_out_deg_hist[deg] += p_neuron_has_out_degree_deg
            out_deg_hist_std[deg] += p_neuron_has_out_degree_deg * (1 - p_neuron_has_out_degree_deg)

            p_neuron_has_in_degree_deg = np.clip(in_degrees_distributions[neuron].pmf[deg], a_min=0, a_max=1)
            average_in_deg_hist[deg] += p_neuron_has_in_degree_deg
            in_deg_hist_std[deg] += p_neuron_has_in_degree_deg * (1 - p_neuron_has_in_degree_deg)
    out_deg_hist_std = np.sqrt(out_deg_hist_std)
    in_deg_hist_std = np.sqrt(in_deg_hist_std)
    return average_in_deg_hist, in_deg_hist_std, average_out_deg_hist, out_deg_hist_std


def calc_reciprocity(graph):
    adj_mat = nx.to_numpy_array(graph)
    num_nodes = graph.number_of_nodes()
    num_reciprocal_dyads = (np.multiply(adj_mat, adj_mat.T)).sum() / 2
    return num_reciprocal_dyads / ((num_nodes * (num_nodes - 1)) / 2)


def calc_norm_triad_motifs_dist(graph):
    norm_dist = np.zeros(NUM_TRIAD_TYPES)
    # No triads are found in a 2 neurons network.
    num_neurons = graph.number_of_nodes()
    if num_neurons < 3:
        return norm_dist
    num_triads = comb(num_neurons, 3)
    census = nx.algorithms.triads.triadic_census(graph)
    i = 0
    for type in SORTED_TRIAD_TYPES:
        norm_dist[i] = census[type] / num_triads
        i += 1
    return norm_dist
