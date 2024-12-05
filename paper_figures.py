import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib
import pylab
import pickle
import os
import numpy as np
import networkx as nx
from sklearn.metrics import roc_auc_score, roc_curve
from matplotlib.ticker import LogFormatter
from tqdm import tqdm
import pandas as pd

from c_elegans_independent_model_training import sample_from_average_adj_mat, calc_model_adj_mat, \
    convert_spls_dict_to_mat, calc_elongation_factor, average_matrix_log_likelihood
from c_elegans_constants import ADULT_WORM_AGE, SINGLE_DEVELOPMENTAL_AGE, FULL_DEVELOPMENTAL_AGES, \
    WORM_LENGTH_NORMALIZATION, THREE_DEVELOPMENTAL_AGES
from er_block_model import generate_er_block_per_type
from CElegansNeuronsAdder import CElegansNeuronsAdder
from wrap_cluster_runs import find_max_likelihood_distance_model, find_max_likelihood_full_model
from graph_statistics import calc_reciprocity, calc_norm_triad_motifs_dist, NUM_TRIAD_TYPES, SORTED_TRIAD_TYPES, \
    average_degree_distribution
from c_elegans_data_parsing import convert_indices_to_names_in_artificial_types, create_cook_types_list, \
    convert_names_to_indices_neuronal_types, construct_exists_array_from_syn_list
from types_partitions_comparison import generate_cluster_intersection_joint_prob_map
from c_elegans_reciprocal_model_training import sample_from_dyads_distribution_train_test, \
    construct_probs_array_from_dyads_dist_string_keys, \
    calc_reciprocal_dependence_model_density_from_dyads_dist_string_keys, \
    calc_reciprocal_dependence_model_log_likelihood_from_dyads_dist_string_keys, \
    calc_reciprocal_dependence_model_variance_from_dyads_dist_str_keys, \
    calc_reciprocal_dependence_model_data_cross_variance_from_dyads_dist_str_keys, RECIPROCAL_DYAD_IDX, \
    ONLY_LOWER_TRIANGLE_SYNAPSE_IDX, ONLY_UPPER_TRIANGLE_SYNAPSE_IDX, \
    calc_reciprocal_dependence_model_dyads_states_distribution, \
    calc_reciprocal_dependence_model_reciprocity_from_dyads_dist, \
    calc_reciprocal_dependence_model_average_adj_mat_from_dyads_distributions_str_keys


def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


FONT_SIZE = 8
MARKER_SIZE = 1
LINE_WIDTH = 1

CM_TO_INCH = 1 / 2.54
SQUARE_FIG_SIZE = (5.467 * CM_TO_INCH, 5.467 * CM_TO_INCH)
RECT_SMALL_FIG_SIZE = (5.467 * CM_TO_INCH, 2 / 3 * 5.467 * CM_TO_INCH)
RECT_MEDIUM_FIG_SIZE = (5.467 * 1.5 * CM_TO_INCH, 5.467 * CM_TO_INCH)
RECT_LARGE_FIG_SIZE = (7.2796 * CM_TO_INCH, 5.467 * CM_TO_INCH)

TYPES_BASE_COLOR = np.array([204 / 255, 204 / 255, 204 / 255])
BIRTH_TIMES_BASE_COLOR = np.array([1.0, 117 / 255, 117 / 255])
DISTANCES_BASE_COLOR = np.array([127 / 255, 158 / 255, 215 / 255])
FULL_MODEL_COLOR = (BIRTH_TIMES_BASE_COLOR + DISTANCES_BASE_COLOR) / 2

SINGLE_EPOCH_INFERRED_COLOR = adjust_lightness('b', 1.5)

NUM_MODEL_SAMPLES_FOR_STATISTICS = 1000

INFERRED_TYPES_INDEX_TO_LABEL = {4: 'C0', 6: 'C1', 7: 'C2', 5: 'C3', 1: 'C4', 0: 'C5', 3: 'C6', 2: 'C7'}
INFERRED_TYPES_LABEL_TO_INDEX = {'C0': 4, 'C1': 6, 'C2': 7, 'C3': 5, 'C4': 1, 'C5': 0, 'C6': 3, 'C7': 2}

COOK_TYPES_COLOR = adjust_lightness('gray', 1.25)


def fig_1_b(out_path="Figures\\Fig1"):
    data_path = "CElegansData\SubTypes\connectomes\\Dataset7.pkl"
    likelihoods_path = 'SavedOutputs\IndependentModel\likelihoods\SubTypes'
    smi, beta, _ = find_max_likelihood_full_model(likelihoods_path)
    spls_path = os.path.join("SavedOutputs\\IndependentModel\S+s\SubTypes", f"spls_smi{smi:.5f}_beta{beta:.5f}.pkl")
    spls_mat, neuronal_types = convert_spls_dict_to_mat(spls_path, 0)
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    neuronal_types_in_nerve_ring = [data.nodes[neuron]['type'] for neuron in data.nodes]
    existing_types_indices = []
    existing_types = []
    for neuronal_type in neuronal_types:
        if neuronal_type in neuronal_types_in_nerve_ring:
            existing_types.append(neuronal_type)
            existing_types_indices.append(neuronal_types.index(neuronal_type))
    existing_types_indices = np.array([existing_types_indices])
    spls_mat = spls_mat[existing_types_indices.T, existing_types_indices]

    max_value = np.log10(spls_mat.max())
    min_value = -5
    fontsize = FONT_SIZE
    main_axes = [0.16, 0.18, 0.65, 0.65]
    colorbar_axes = [0.82, 0.18, 0.025, 0.65]
    pylab.rcParams['xtick.major.pad'] = '0.5'
    pylab.rcParams['ytick.major.pad'] = '0.5'
    axis_labelpad = 1
    colorbar_labelpad = 3
    fig = plt.figure(figsize=SQUARE_FIG_SIZE)
    ax = fig.add_axes(main_axes)
    im = ax.imshow(spls_mat, cmap="Greens", norm=colors.LogNorm(vmin=10 ** min_value, vmax=10 ** max_value))
    cbar_ax = fig.add_axes(colorbar_axes)
    majorticks = np.logspace(-5, -2, num=4)
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=majorticks)
    minorticks = []
    for i in range(3, 6):
        minorticks += list(np.arange(2, 11) / 10 ** i)
    for j in range(2, 11):
        if j / 10 ** 2 > spls_mat.max():
            break
        minorticks.append(j / 10 ** 2)
    cbar.ax.yaxis.set_ticks(minorticks, minor=True)
    cbar.ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    cbar_ax.set_yticklabels([r"$10^{{{0:d}}}$".format(-i) for i in range(5, 1, -1)],
                            fontsize=fontsize)
    cbar.ax.set_title('$S^{+}$', fontsize=fontsize, pad=colorbar_labelpad, loc='left')
    ax.set_ylabel('pre-synaptic cell type', fontsize=fontsize, labelpad=axis_labelpad)
    ax.set_xlabel('post-synaptic cell type', fontsize=fontsize, labelpad=axis_labelpad)
    ax.set_xticks(np.arange(len(existing_types)))
    ax.set_xticklabels(existing_types, fontsize=fontsize, rotation=270)
    ax.set_yticks(np.arange(len(existing_types)))
    ax.set_yticklabels(existing_types, fontsize=fontsize)
    plt.savefig(os.path.join(out_path, '1_b.pdf'), format='pdf')
    plt.show()


def fig_1_c(out_path="Figures\\Fig1"):
    likelihoods_path = 'SavedOutputs\IndependentModel\likelihoods\SubTypes'
    _, beta, _ = find_max_likelihood_full_model(likelihoods_path)
    distances = np.arange(0, 1001, 1)
    decay = np.exp(-beta * calc_elongation_factor(ADULT_WORM_AGE) * distances / WORM_LENGTH_NORMALIZATION)

    fontsize = FONT_SIZE
    main_axes = [0.18, 0.18, 0.65, 0.65]
    pylab.rcParams['xtick.major.pad'] = '0.5'
    pylab.rcParams['ytick.major.pad'] = '0.5'
    axis_labelpad = 1

    fig = plt.figure(figsize=SQUARE_FIG_SIZE)
    ax1 = fig.add_axes(main_axes)
    xticks = np.arange(0, 1001, 250)
    yticks = np.arange(0.5, 1.1, 0.25)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels([f'{int(tick)}' if int(tick) not in [250, 750] else '' for tick in xticks], fontsize=fontsize)
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(['0.5', '', '1.0'], fontsize=fontsize)
    ax1.set_xlabel('distance between neurons [$\mu m$]', fontsize=fontsize, labelpad=axis_labelpad)
    ax1.set_ylabel('synaptic formation probability\ndecay factor', fontsize=fontsize, labelpad=axis_labelpad - 7)
    ax1.plot(distances, decay, color='g')
    plt.savefig(os.path.join(out_path, "1_c.pdf"), format='pdf')
    plt.show()


def fig_1_d_e_f(out_path="Figures\\Fig1"):
    np.random.seed(123456789)

    data_connectome_path = 'CElegansData\SubTypes\\connectomes\Dataset8.pkl'
    with open(data_connectome_path, 'rb') as f:
        data_connectome = pickle.load(f)
    alphabetic_neuronal_names = sorted(data_connectome.nodes)
    data_adj_mat = nx.to_numpy_array(data_connectome, nodelist=alphabetic_neuronal_names)

    neurons_list_path = "CElegansData\\nerve_ring_neurons_subset.pkl"
    cook_types_list, _ = create_cook_types_list(neurons_list_path)
    neurons_names_by_type = []
    for n_type in cook_types_list:
        neurons_names_by_type += sorted(list(n_type))

    neurons_idx_by_type = np.zeros((len(alphabetic_neuronal_names), 1)).astype(int)
    for j in range(len(alphabetic_neuronal_names)):
        neurons_idx_by_type[j] = alphabetic_neuronal_names.index(neurons_names_by_type[j])

    data_adj_mat = data_adj_mat[neurons_idx_by_type, neurons_idx_by_type.T]

    likelihoods_path = 'SavedOutputs\IndependentModel\likelihoods\SubTypes'
    smi, beta, _ = find_max_likelihood_full_model(likelihoods_path)
    average_adj_mat_path = os.path.join("SavedOutputs\IndependentModel\\average_adj_mats\SubTypes",
                                        f"smi{smi:.5f}_beta{beta:.5f}_adult.pkl")
    with open(average_adj_mat_path, 'rb') as f:
        model_average_adj_mat = pickle.load(f)

    model_single_draw = sample_from_average_adj_mat(model_average_adj_mat)

    model_average_adj_mat = model_average_adj_mat[neurons_idx_by_type, neurons_idx_by_type.T]
    model_single_draw = model_single_draw[neurons_idx_by_type, neurons_idx_by_type.T]

    axis_ticks = range(0, 181, 60)
    fontsize = FONT_SIZE
    main_axes = [0.18, 0.18, 0.65, 0.65]
    colorbar_axes = [0.85, 0.18, 0.03, 0.65]
    pylab.rcParams['xtick.major.pad'] = '0.5'
    pylab.rcParams['ytick.major.pad'] = '0.5'
    axis_labelpad = 1
    colorbar_labelpad = -5
    dy = 0.5

    data_cmap = colors.LinearSegmentedColormap.from_list('data', [(1, 1, 1), (0, 0, 0)])
    model_cmap = colors.LinearSegmentedColormap.from_list('model', [(1, 1, 1), FULL_MODEL_COLOR])

    fig1 = plt.figure(1, figsize=SQUARE_FIG_SIZE)
    ax1 = fig1.add_axes(main_axes)
    x0, x1 = ax1.get_xlim()
    y0, y1 = ax1.get_ylim()
    ax1.set_aspect(abs(x1 - x0) / abs(y1 - y0), 'box')
    im1 = ax1.imshow(data_adj_mat, cmap=data_cmap)
    ax1.set_xlabel("post-synaptic neuronal idx", fontsize=fontsize, labelpad=axis_labelpad)
    ax1.set_ylabel("pre-synaptic neuronal idx", fontsize=fontsize, labelpad=axis_labelpad)
    ax1.set_xticks(axis_ticks)
    ax1.set_xticklabels([str(i) for i in axis_ticks], fontsize=fontsize)
    ax1.set_yticks(axis_ticks)
    ax1.set_yticklabels([str(i) for i in axis_ticks], fontsize=fontsize)
    fig1.savefig(os.path.join(out_path, '1_d.pdf'), format='pdf')
    plt.show()

    fig2 = plt.figure(2, figsize=SQUARE_FIG_SIZE)
    ax2 = fig2.add_axes(main_axes)
    x0, x1 = ax2.get_xlim()
    y0, y1 = ax2.get_ylim()
    ax2.set_aspect(abs(x1 - x0) / abs(y1 - y0), 'box')
    im2 = ax2.imshow(model_single_draw, cmap=model_cmap)
    ax2.set_xlabel("post-synaptic neuronal idx", fontsize=fontsize, labelpad=axis_labelpad)
    ax2.set_ylabel("pre-synaptic neuronal idx", fontsize=fontsize, labelpad=axis_labelpad)
    ax2.set_xticks(axis_ticks)
    ax2.set_xticklabels([str(i) for i in axis_ticks], fontsize=fontsize)
    ax2.set_yticks(axis_ticks)
    ax2.set_yticklabels([str(i) for i in axis_ticks], fontsize=fontsize)
    fig2.savefig(os.path.join(out_path, '1_e.pdf'), format='pdf')
    plt.show()

    fig3 = plt.figure(3, figsize=SQUARE_FIG_SIZE)
    ax3 = fig3.add_axes(main_axes)
    x0, x1 = ax3.get_xlim()
    y0, y1 = ax3.get_ylim()
    ax3.set_aspect(abs(x1 - x0) / abs(y1 - y0), 'box')
    im3 = ax3.imshow(model_average_adj_mat, cmap=model_cmap)
    ax3.set_xlabel("post-synaptic neuronal idx", fontsize=fontsize, labelpad=axis_labelpad)
    ax3.set_ylabel("pre-synaptic neuronal idx", fontsize=fontsize, labelpad=axis_labelpad)
    ax3.set_xticks(axis_ticks)
    ax3.set_xticklabels([str(i) for i in axis_ticks], fontsize=fontsize)
    ax3.set_yticks(axis_ticks)
    ax3.set_yticklabels([str(i) for i in axis_ticks], fontsize=fontsize)
    cbar_ax = fig3.add_axes(colorbar_axes)
    cbar3 = fig3.colorbar(im3, cax=cbar_ax,
                          ticks=[0, 0.6 / 4, 3 * 0.6 / 4,
                                 0.6])
    cbar_ax.set_yticklabels(
        ['0', '', '', '0.6'],
        fontsize=fontsize)
    cbar3.set_label('probability', rotation=270, labelpad=colorbar_labelpad, y=dy, fontsize=fontsize)
    fig3.savefig(os.path.join(out_path, '1_f.pdf'), format='pdf')
    plt.show()


def fig_1_g(out_path="Figures\\Fig1"):
    data_connectome_path = 'CElegansData\SubTypes\connectomes\Dataset8.pkl'
    with open(data_connectome_path, 'rb') as f:
        data_connectome = pickle.load(f)
    data_adj_mat = nx.to_numpy_array(data_connectome, nodelist=sorted(data_connectome.nodes))
    data_adj_mat = data_adj_mat.astype(int)

    likelihoods_path = 'SavedOutputs\IndependentModel\likelihoods\SubTypes'
    smi, beta, _ = find_max_likelihood_full_model(likelihoods_path)
    model_average_adj_mat_path = f"SavedOutputs\IndependentModel\\average_adj_mats\SubTypes\\smi{smi:.5f}_beta{beta:.5f}_adult.pkl"
    with open(model_average_adj_mat_path, 'rb') as f:
        model_average_adj_mat = pickle.load(f)

    false_positive_rate, true_positive_rate, _ = roc_curve(data_adj_mat.flatten(), model_average_adj_mat.flatten())
    auc = roc_auc_score(data_adj_mat.flatten(), model_average_adj_mat.flatten())

    fig = plt.figure(figsize=SQUARE_FIG_SIZE)
    fontsize = FONT_SIZE
    markersize = MARKER_SIZE
    line_width = LINE_WIDTH
    main_axes = [0.18, 0.18, 0.65, 0.65]
    pylab.rcParams['xtick.major.pad'] = '0.5'
    pylab.rcParams['ytick.major.pad'] = '0.5'
    axis_labelpad = 1

    ax1 = fig.add_axes(main_axes)
    ax1.set_aspect('equal', 'box')
    axes_ticks = np.arange(0, 1.2, 0.5)
    ax1.set_xticks(axes_ticks)
    ax1.set_xticklabels([f'{tick:.1f}' for tick in axes_ticks], fontsize=fontsize)
    ax1.set_yticks(axes_ticks)
    ax1.set_yticklabels([f'{tick:.1f}' for tick in axes_ticks], fontsize=fontsize)
    ax1.plot([0, 1], [0, 1], 'gray', lw=line_width)
    ax1.set_xlabel('False Positive Rate', fontsize=fontsize, labelpad=axis_labelpad)
    ax1.set_ylabel('True Positive Rate', fontsize=fontsize, labelpad=axis_labelpad)

    single_color = FULL_MODEL_COLOR

    ax1.plot(false_positive_rate,
             true_positive_rate, marker='.', label=f"AUC={auc:.2f}", c=single_color,
             markersize=markersize, lw=line_width)
    fig.savefig(os.path.join(out_path, '1_g.pdf'), format='pdf')
    plt.show()


def fig_2_b(out_path="Figures\\Fig2", saved_calcs_path="Figures\SavedCalcs\\average_mats", is_saved=False,
            do_save=True):
    data_connectome_path = 'CElegansData\SubTypes\connectomes\Dataset8.pkl'
    with open(data_connectome_path, 'rb') as f:
        data_connectome = pickle.load(f)
    data_adj_mat = nx.to_numpy_array(data_connectome, nodelist=sorted(data_connectome.nodes))
    data_adj_mat = data_adj_mat.astype(int)

    if not is_saved:
        _, types_single_average_mat = generate_er_block_per_type(CElegansNeuronsAdder.SINGLE_TYPE,
                                                                 'SingleType\connectomes\Dataset7.pkl',
                                                                 'CElegansData\\nerve_ring_neurons_subset.pkl')

        birth_times_beta = 0
        smi_single_types_birth_times, _, _ = find_max_likelihood_full_model(
            "SavedOutputs\IndependentModel\likelihoods\SingleType",
            beta_value=birth_times_beta)
        birth_times_single_type_spls_path = f"SavedOutputs\IndependentModel\S+s\SingleType\\spls_smi{smi_single_types_birth_times:.5f}_beta{birth_times_beta:.5f}.pkl"
        with open(birth_times_single_type_spls_path, 'rb') as f:
            birth_times_single_type_spls = pickle.load(f)
        birth_times_single_type_average_mat = calc_model_adj_mat(birth_times_single_type_spls,
                                                                 smi_single_types_birth_times,
                                                                 birth_times_beta, ADULT_WORM_AGE,
                                                                 SINGLE_DEVELOPMENTAL_AGE,
                                                                 'CElegansData\SingleType\connectomes\Dataset7.pkl')

        smi_single_type_full, beta_single_type_full, _ = find_max_likelihood_full_model(
            "SavedOutputs\IndependentModel\likelihoods\SingleType")
        full_single_type_spls_path = f"SavedOutputs\IndependentModel\S+s\SingleType\\spls_smi{smi_single_type_full:.5f}_beta{beta_single_type_full:.5f}.pkl"
        with open(full_single_type_spls_path, 'rb') as f:
            full_single_type_spls = pickle.load(f)
        full_single_type_average_mat = calc_model_adj_mat(full_single_type_spls, smi_single_type_full,
                                                          beta_single_type_full, ADULT_WORM_AGE,
                                                          SINGLE_DEVELOPMENTAL_AGE,
                                                          'CElegansData\SingleType\\connectomes\Dataset7.pkl')

        if do_save:
            with open(os.path.join(saved_calcs_path, 'types_1_types_average.pkl'), 'wb') as f:
                pickle.dump(types_single_average_mat, f)

            with open(os.path.join(saved_calcs_path, "birth_times_1_types_average.pkl"), 'wb') as f:
                pickle.dump(birth_times_single_type_average_mat, f)

            with open(os.path.join(saved_calcs_path, 'full_1_types_average.pkl'), 'wb') as f:
                pickle.dump(full_single_type_average_mat, f)
    else:
        with open(os.path.join(saved_calcs_path, 'types_1_types_average.pkl'), 'rb') as f:
            types_single_average_mat = pickle.load(f)

        with open(os.path.join(saved_calcs_path, "birth_times_1_types_average.pkl"), 'rb') as f:
            birth_times_single_type_average_mat = pickle.load(f)

        with open(os.path.join(saved_calcs_path, 'full_1_types_average.pkl'), 'rb') as f:
            full_single_type_average_mat = pickle.load(f)

    types_single_false_positive, types_single_true_positive, _ = roc_curve(data_adj_mat.flatten(),
                                                                           types_single_average_mat.flatten())

    beta_single_type_distances, _ = find_max_likelihood_distance_model(
        "SavedOutputs\DistancesModel\likelihoods\SingleType")
    single_type_distances_average_mat_path = f"SavedOutputs\DistancesModel\\average_adj_mats\SingleType\\{beta_single_type_distances:.3f}_average_adj_mat.pkl"
    with open(single_type_distances_average_mat_path, 'rb') as f:
        single_type_distances_average_mat = pickle.load(f)
    distance_single_false_positive, distance_single_true_positive, _ = roc_curve(data_adj_mat.flatten(),
                                                                                 single_type_distances_average_mat.flatten())

    birth_times_single_false_positive, birth_times_single_true_positive, _ = roc_curve(data_adj_mat.flatten(),
                                                                                       birth_times_single_type_average_mat.flatten())

    full_single_false_positive, full_single_true_positive, _ = roc_curve(data_adj_mat.flatten(),
                                                                         full_single_type_average_mat.flatten())

    fontsize = FONT_SIZE
    markersize = MARKER_SIZE
    line_width = LINE_WIDTH
    axes_labelpad = 1
    main_axes = [0.2, 0.175, 0.78, 0.78]
    fig = plt.figure(figsize=SQUARE_FIG_SIZE)
    ax1 = fig.add_axes(main_axes)
    ax1.set_aspect('equal', 'box')
    axes_ticks = np.arange(0, 1.2, 0.5)
    ax1.set_xticks(axes_ticks)
    ax1.set_xticklabels([f'{tick:.1f}' for tick in axes_ticks], fontsize=fontsize)
    ax1.set_yticks(axes_ticks)
    ax1.set_yticklabels([f'{tick:.1f}' for tick in axes_ticks], fontsize=fontsize)
    ax1.plot([0, 1], [0, 1], 'gray', lw=line_width)
    ax1.set_xlabel('False Positive Rate', fontsize=fontsize, labelpad=axes_labelpad)
    ax1.set_ylabel('True Positive Rate', fontsize=fontsize, labelpad=axes_labelpad)

    birth_times_single_color = BIRTH_TIMES_BASE_COLOR
    ax1.plot(birth_times_single_false_positive, birth_times_single_true_positive, marker='.',
             color=tuple(birth_times_single_color),
             lw=line_width, markersize=markersize)
    distance_single_color = 'indigo'
    ax1.plot(distance_single_false_positive, distance_single_true_positive, marker='.',
             color=distance_single_color,
             lw=line_width, markersize=markersize)
    plt.savefig(os.path.join(out_path, '2_b.pdf'), format='pdf')
    plt.show()


def fig_2_c(out_path="Figures\\Fig2", saved_calcs_path="Figures\SavedCalcs\\average_mats", is_saved=False,
            do_save=True):
    data_connectome_path = 'CElegansData\SubTypes\\connectomes\Dataset8.pkl'
    with open(data_connectome_path, 'rb') as f:
        data_connectome = pickle.load(f)
    data_adj_mat = nx.to_numpy_array(data_connectome, nodelist=sorted(data_connectome.nodes))
    data_adj_mat = data_adj_mat.astype(int)

    if not is_saved:
        smi_single_type_full, beta_single_type_full, _ = find_max_likelihood_full_model(
            "SavedOutputs\IndependentModel\likelihoods\SingleType")
        full_single_type_spls_path = f"SavedOutputs\IndependentModel\S+s\SingleType\\spls_smi{smi_single_type_full:.5f}_beta{beta_single_type_full:.5f}.pkl"
        with open(full_single_type_spls_path, 'rb') as f:
            full_single_type_spls = pickle.load(f)
        full_single_type_average_mat = calc_model_adj_mat(full_single_type_spls, smi_single_type_full,
                                                          beta_single_type_full, ADULT_WORM_AGE,
                                                          SINGLE_DEVELOPMENTAL_AGE,
                                                          'CElegansData\SingleType\\connectomes\Dataset7.pkl')

        smi_coarse_types_full, beta_coarse_types_full, _ = find_max_likelihood_full_model(
            "SavedOutputs\IndependentModel\likelihoods\CoarseTypes")
        full_coarse_types_spls_path = f"SavedOutputs\IndependentModel\S+s\CoarseTypes\\spls_smi{smi_coarse_types_full:.5f}_beta{beta_coarse_types_full:.5f}.pkl"
        with open(full_coarse_types_spls_path, 'rb') as f:
            full_coarse_types_spls = pickle.load(f)
        full_coarse_types_average_mat = calc_model_adj_mat(full_coarse_types_spls, smi_coarse_types_full,
                                                           beta_coarse_types_full, ADULT_WORM_AGE,
                                                           SINGLE_DEVELOPMENTAL_AGE,
                                                           'CElegansData\CoarseTypes\\connectomes\Dataset7.pkl')

        smi_sub_types_full, beta_sub_types_full, _ = find_max_likelihood_full_model(
            "SavedOutputs\IndependentModel\likelihoods\SubTypes")
        full_sub_types_spls_path = f"SavedOutputs\IndependentModel\S+s\SubTypes\\spls_smi{smi_sub_types_full:.5f}_beta{beta_sub_types_full:.5f}.pkl"
        with open(full_sub_types_spls_path, 'rb') as f:
            full_sub_types_spls = pickle.load(f)
        full_sub_types_average_mat = calc_model_adj_mat(full_sub_types_spls, smi_sub_types_full,
                                                        beta_sub_types_full, ADULT_WORM_AGE,
                                                        SINGLE_DEVELOPMENTAL_AGE,
                                                        'CElegansData\SubTypes\\connectomes\Dataset7.pkl')
        if do_save:
            with open(os.path.join(saved_calcs_path, 'full_1_types_average.pkl'), 'wb') as f:
                pickle.dump(full_single_type_average_mat, f)
            with open(os.path.join(saved_calcs_path, "full_coarse_types_average.pkl"), 'wb') as f:
                pickle.dump(full_coarse_types_average_mat, f)
            with open(os.path.join(saved_calcs_path, "full_sub_types_average.pkl"), 'wb') as f:
                pickle.dump(full_sub_types_average_mat, f)
    else:
        with open(os.path.join(saved_calcs_path, 'full_1_types_average.pkl'), 'rb') as f:
            full_single_type_average_mat = pickle.load(f)
        with open(os.path.join(saved_calcs_path, "full_coarse_types_average.pkl"), 'rb') as f:
            full_coarse_types_average_mat = pickle.load(f)
        with open(os.path.join(saved_calcs_path, "full_sub_types_average.pkl"), 'rb') as f:
            full_sub_types_average_mat = pickle.load(f)

    full_single_false_positive, full_single_true_positive, _ = roc_curve(data_adj_mat.flatten(),
                                                                         full_single_type_average_mat.flatten())

    full_coarse_false_positive, full_coarse_true_positive, _ = roc_curve(data_adj_mat.flatten(),
                                                                         full_coarse_types_average_mat.flatten())

    full_sub_false_positive, full_sub_true_positive, _ = roc_curve(data_adj_mat.flatten(),
                                                                   full_sub_types_average_mat.flatten())

    fontsize = FONT_SIZE
    markersize = MARKER_SIZE
    line_width = LINE_WIDTH
    axes_labelpad = 1
    main_axes = [0.2, 0.175, 0.78, 0.78]
    fig = plt.figure(figsize=SQUARE_FIG_SIZE)
    ax1 = fig.add_axes(main_axes)
    ax1.set_aspect('equal', 'box')
    axes_ticks = np.arange(0, 1.2, 0.5)
    ax1.set_xticks(axes_ticks)
    ax1.set_xticklabels([f'{tick:.1f}' for tick in axes_ticks], fontsize=fontsize)
    ax1.set_yticks(axes_ticks)
    ax1.set_yticklabels([f'{tick:.1f}' for tick in axes_ticks], fontsize=fontsize)
    ax1.plot([0, 1], [0, 1], 'gray', lw=line_width)
    ax1.set_xlabel('False Positive Rate', fontsize=fontsize, labelpad=axes_labelpad)
    ax1.set_ylabel('True Positive Rate', fontsize=fontsize, labelpad=axes_labelpad)

    full_single_color = adjust_lightness((BIRTH_TIMES_BASE_COLOR + DISTANCES_BASE_COLOR) / 2, 1.3)
    ax1.plot(full_single_false_positive, full_single_true_positive, marker='.', color=tuple(full_single_color),
             lw=line_width, markersize=markersize)

    full_coarse_color = adjust_lightness((BIRTH_TIMES_BASE_COLOR + DISTANCES_BASE_COLOR) / 2, 1.0)
    ax1.plot(full_coarse_false_positive, full_coarse_true_positive, marker='.', color=tuple(full_coarse_color),
             lw=line_width, markersize=markersize)
    full_sub_color = adjust_lightness((BIRTH_TIMES_BASE_COLOR + DISTANCES_BASE_COLOR) / 2, 0.7)
    ax1.plot(full_sub_false_positive, full_sub_true_positive, marker='.', color=tuple(full_sub_color), lw=line_width,
             markersize=markersize)

    plt.savefig(os.path.join(out_path, '2_c.pdf'), format='pdf')
    plt.show()


def fig_2_d(out_path="Figures\\Fig2", saved_calcs_path="Figures\SavedCalcs\\average_mats", is_saved=False,
            do_save=True):
    data_connectome_path = 'CElegansData\SubTypes\\connectomes\Dataset8.pkl'
    with open(data_connectome_path, 'rb') as f:
        data_connectome = pickle.load(f)
    data_adj_mat = nx.to_numpy_array(data_connectome, nodelist=sorted(data_connectome.nodes))
    data_adj_mat = data_adj_mat.astype(int)

    num_types_range = range(1, 14)

    if not is_saved:
        smi_single_type_full, beta_single_type_full, _ = find_max_likelihood_full_model(
            "SavedOutputs\IndependentModel\likelihoods\SingleType")
        full_single_type_spls_path = f"SavedOutputs\IndependentModel\S+s\SingleType\\spls_smi{smi_single_type_full:.5f}_beta{beta_single_type_full:.5f}.pkl"
        with open(full_single_type_spls_path, 'rb') as f:
            full_single_type_spls = pickle.load(f)
        full_single_type_average_mat = calc_model_adj_mat(full_single_type_spls, smi_single_type_full,
                                                          beta_single_type_full, ADULT_WORM_AGE,
                                                          SINGLE_DEVELOPMENTAL_AGE,
                                                          'CElegansData\SingleType\\connectomes\Dataset7.pkl')

        smi_coarse_types_full, beta_coarse_types_full, _ = find_max_likelihood_full_model(
            "SavedOutputs\IndependentModel\likelihoods\CoarseTypes")
        full_coarse_types_spls_path = f"SavedOutputs\IndependentModel\S+s\CoarseTypes\\spls_smi{smi_coarse_types_full:.5f}_beta{beta_coarse_types_full:.5f}.pkl"
        with open(full_coarse_types_spls_path, 'rb') as f:
            full_coarse_types_spls = pickle.load(f)
        full_coarse_types_average_mat = calc_model_adj_mat(full_coarse_types_spls, smi_coarse_types_full,
                                                           beta_coarse_types_full, ADULT_WORM_AGE,
                                                           SINGLE_DEVELOPMENTAL_AGE,
                                                           'CElegansData\CoarseTypes\\connectomes\Dataset7.pkl')

        smi_sub_types_full, beta_sub_types_full, _ = find_max_likelihood_full_model(
            "SavedOutputs\IndependentModel\likelihoods\SubTypes")
        full_sub_types_spls_path = f"SavedOutputs\IndependentModel\S+s\SubTypes\\spls_smi{smi_sub_types_full:.5f}_beta{beta_sub_types_full:.5f}.pkl"
        with open(full_sub_types_spls_path, 'rb') as f:
            full_sub_types_spls = pickle.load(f)
        full_sub_types_average_mat = calc_model_adj_mat(full_sub_types_spls, smi_sub_types_full,
                                                        beta_sub_types_full, ADULT_WORM_AGE,
                                                        SINGLE_DEVELOPMENTAL_AGE,
                                                        'CElegansData\SubTypes\\connectomes\Dataset7.pkl')

        average_mats_inferred_types = []
        for num_types in num_types_range:
            smi_full, beta_full, _ = find_max_likelihood_full_model(
                f"SavedOutputs\IndependentModel\likelihoods\InferredTypes\\{num_types}_types")
            full_spls_path = f"SavedOutputs\IndependentModel\S+s\InferredTypes\\{num_types}_types\\spls_smi{smi_full:.5f}_beta{beta_full:.5f}.pkl"
            with open(full_spls_path, 'rb') as f:
                full_spls = pickle.load(f)
            full_average_mat = calc_model_adj_mat(full_spls, smi_full,
                                                  beta_full, ADULT_WORM_AGE,
                                                  SINGLE_DEVELOPMENTAL_AGE,
                                                  f'CElegansData\InferredTypes\\connectomes\\{num_types}_types\Dataset7.pkl')
            average_mats_inferred_types.append(full_average_mat)

        average_mats_random_types = []
        for num_types in num_types_range:
            smi_full, beta_full, _ = find_max_likelihood_full_model(
                f"SavedOutputs\IndependentModel\likelihoods\RandomTypes\\{num_types}_types")
            full_spls_path = f"SavedOutputs\IndependentModel\S+s\RandomTypes\\{num_types}_types\\spls_smi{smi_full:.5f}_beta{beta_full:.5f}.pkl"
            with open(full_spls_path, 'rb') as f:
                full_spls = pickle.load(f)
            full_average_mat = calc_model_adj_mat(full_spls, smi_full,
                                                  beta_full, ADULT_WORM_AGE,
                                                  SINGLE_DEVELOPMENTAL_AGE,
                                                  f'CElegansData\RandomTypes\\connectomes\\{num_types}_types\Dataset7.pkl')
            average_mats_random_types.append(full_average_mat)

        if do_save:
            with open(os.path.join(saved_calcs_path, 'full_1_types_average.pkl'), 'wb') as f:
                pickle.dump(full_single_type_average_mat, f)
            with open(os.path.join(saved_calcs_path, "full_coarse_types_average.pkl"), 'wb') as f:
                pickle.dump(full_coarse_types_average_mat, f)
            with open(os.path.join(saved_calcs_path, "full_sub_types_average.pkl"), 'wb') as f:
                pickle.dump(full_sub_types_average_mat, f)

            for num_types in num_types_range:
                with open(os.path.join(saved_calcs_path, f'full_{num_types}_types_average.pkl'), 'wb') as f:
                    pickle.dump(average_mats_inferred_types[num_types - 1], f)
                with open(os.path.join(saved_calcs_path, f'full_{num_types}_random_types_average.pkl'), 'wb') as f:
                    pickle.dump(average_mats_random_types[num_types - 1], f)
    else:
        with open(os.path.join(saved_calcs_path, 'full_1_types_average.pkl'), 'rb') as f:
            full_single_type_average_mat = pickle.load(f)
        with open(os.path.join(saved_calcs_path, "full_coarse_types_average.pkl"), 'rb') as f:
            full_coarse_types_average_mat = pickle.load(f)
        with open(os.path.join(saved_calcs_path, "full_sub_types_average.pkl"), 'rb') as f:
            full_sub_types_average_mat = pickle.load(f)

        average_mats_inferred_types = []
        average_mats_random_types = []
        for num_types in num_types_range:
            with open(os.path.join(saved_calcs_path, f'full_{num_types}_types_average.pkl'), 'rb') as f:
                average_mats_inferred_types.append(pickle.load(f))
            with open(os.path.join(saved_calcs_path, f'full_{num_types}_random_types_average.pkl'), 'rb') as f:
                average_mats_random_types.append(pickle.load(f))

    full_single_auc = roc_auc_score(data_adj_mat.flatten(), full_single_type_average_mat.flatten())
    full_coarse_auc = roc_auc_score(data_adj_mat.flatten(), full_coarse_types_average_mat.flatten())
    full_sub_auc = roc_auc_score(data_adj_mat.flatten(), full_sub_types_average_mat.flatten())

    full_model_aucs = np.zeros(len(num_types_range))
    full_model_random_types_aucs = np.zeros((len(num_types_range)))
    for num_types in num_types_range:
        full_average_mat = average_mats_inferred_types[num_types - 1]
        full_model_aucs[num_types - 1] = roc_auc_score(data_adj_mat.flatten(), full_average_mat.flatten())

        full_average_mat_random_types = average_mats_random_types[num_types - 1]
        full_model_random_types_aucs[num_types - 1] = roc_auc_score(data_adj_mat.flatten(),
                                                                    full_average_mat_random_types.flatten())

    fig = plt.figure(figsize=SQUARE_FIG_SIZE)
    fontsize = FONT_SIZE
    line_width = LINE_WIDTH
    markersize = MARKER_SIZE * 3
    pylab.rcParams['xtick.major.pad'] = '0.5'
    pylab.rcParams['ytick.major.pad'] = '0.5'
    axis_labelpad = 1
    main_axes = [0.2, 0.175, 0.78, 0.78]

    ax1 = fig.add_axes(main_axes)
    x_axes_ticks = np.arange(1, 14, 4)
    y_axes_ticks = np.arange(0.5, 1, 0.1)
    ax1.set_xticks(x_axes_ticks)
    ax1.set_xticklabels([f'{tick}' for tick in x_axes_ticks], fontsize=fontsize)
    ax1.set_yticks(y_axes_ticks)
    ax1.set_yticklabels([f'{tick:.1f}' for tick in y_axes_ticks], fontsize=fontsize)
    ax1.set_xlabel('number of cell types', fontsize=fontsize, labelpad=axis_labelpad)
    ax1.set_ylabel('AUROC', fontsize=fontsize, labelpad=axis_labelpad)
    ax1.set_xlim(0, 14)
    ax1.set_ylim(0.5, 0.85)

    ax1.plot(num_types_range, full_model_random_types_aucs, marker='.', c='dimgray', lw=line_width,
             markersize=markersize,
             label='random types')
    ax1.plot([1, 3, 13], [full_single_auc, full_coarse_auc, full_sub_auc], marker='.',
             c=(BIRTH_TIMES_BASE_COLOR + DISTANCES_BASE_COLOR) / 2, lw=line_width,
             markersize=markersize, label='functional types')
    ax1.plot(num_types_range, full_model_aucs, marker='.',
             c=SINGLE_EPOCH_INFERRED_COLOR, lw=line_width,
             markersize=markersize, label='inferred types')
    plt.savefig(os.path.join(out_path, '2_d.pdf'), format='pdf')
    plt.show()


def _get_neurons_idx_by_inferred_type(num_types):
    neurons_list_path = os.path.join("CElegansData", "nerve_ring_neurons_subset.pkl")
    with open(neurons_list_path, 'rb') as f:
        nerve_ring_neurons = pickle.load(f)
    alphabetic_neuronal_names = sorted(nerve_ring_neurons)
    neuronal_types_path = os.path.join("CElegansData", "InferredTypes", "types", f"{num_types}.pkl")

    neuronal_types_by_names = convert_indices_to_names_in_artificial_types(neuronal_types_path, neurons_list_path)

    neuronal_names_ordered_by_type = []
    for neuronal_type in neuronal_types_by_names:
        neuronal_names_ordered_by_type += sorted(list(neuronal_type))

    neurons_idx_by_type = np.zeros((len(alphabetic_neuronal_names), 1)).astype(int)
    for j in range(len(alphabetic_neuronal_names)):
        neurons_idx_by_type[j] = alphabetic_neuronal_names.index(neuronal_names_ordered_by_type[j])
    return neurons_idx_by_type


def fig_2_e_f_g(out_path="Figures\\Fig2"):
    np.random.seed(123456789)
    num_types = 8
    train_data_connectome_path = 'CElegansData\SubTypes\\connectomes\Dataset7.pkl'
    with open(train_data_connectome_path, 'rb') as f:
        train_data_connectome = pickle.load(f)
    train_data_adj_mat = nx.to_numpy_array(train_data_connectome, nodelist=sorted(list(train_data_connectome.nodes)))

    neurons_idx_by_type = _get_neurons_idx_by_inferred_type(num_types)

    train_data_adj_mat = train_data_adj_mat[neurons_idx_by_type, neurons_idx_by_type.T]

    data_connectome_path = 'CElegansData\SubTypes\\connectomes\Dataset8.pkl'
    with open(data_connectome_path, 'rb') as f:
        data_connectome = pickle.load(f)
    alphabetic_neuronal_names = sorted(data_connectome.nodes)
    data_adj_mat = nx.to_numpy_array(data_connectome, nodelist=alphabetic_neuronal_names)

    data_adj_mat = data_adj_mat[neurons_idx_by_type, neurons_idx_by_type.T]

    model_adj_mat_path = f"SavedOutputs\IndependentModel\\average_adj_mats\InferredTypes\\{num_types}_types.pkl"
    with open(model_adj_mat_path, 'rb') as f:
        model_average_adj_mat = pickle.load(f)

    model_single_draw = sample_from_average_adj_mat(model_average_adj_mat)

    model_single_draw = model_single_draw[neurons_idx_by_type, neurons_idx_by_type.T]

    data_cmap = colors.LinearSegmentedColormap.from_list('data', [(1, 1, 1), (0, 0, 0)])
    single_color = SINGLE_EPOCH_INFERRED_COLOR
    model_cmap = colors.LinearSegmentedColormap.from_list('single_epoch', [(1, 1, 1), single_color])
    axis_ticks = range(0, 181, 60)
    fig_size = SQUARE_FIG_SIZE
    fontsize = FONT_SIZE
    main_axes = [0.18, 0.18, 0.75, 0.75]
    pylab.rcParams['xtick.major.pad'] = '0.5'
    pylab.rcParams['ytick.major.pad'] = '0.5'
    axis_labelpad = 1

    fig0 = plt.figure(0, figsize=fig_size)
    ax0 = fig0.add_axes(main_axes)
    x0, x1 = ax0.get_xlim()
    y0, y1 = ax0.get_ylim()
    ax0.set_aspect(abs(x1 - x0) / abs(y1 - y0), 'box')
    im0 = ax0.imshow(train_data_adj_mat, cmap=data_cmap)
    ax0.set_xlabel("post-synaptic neuronal idx", fontsize=fontsize, labelpad=axis_labelpad)
    ax0.set_ylabel("pre-synaptic neuronal idx", fontsize=fontsize, labelpad=axis_labelpad)
    ax0.set_xticks(axis_ticks)
    ax0.set_xticklabels([str(i) for i in axis_ticks], fontsize=fontsize)
    ax0.set_yticks(axis_ticks)
    ax0.set_yticklabels([str(i) for i in axis_ticks], fontsize=fontsize)
    fig0.savefig(os.path.join(out_path, '2_e_worm_7.pdf'), format='pdf')
    plt.show()

    fig1 = plt.figure(1, figsize=fig_size)
    ax1 = fig1.add_axes(main_axes)
    x0, x1 = ax1.get_xlim()
    y0, y1 = ax1.get_ylim()
    ax1.set_aspect(abs(x1 - x0) / abs(y1 - y0), 'box')
    im1 = ax1.imshow(data_adj_mat, cmap=data_cmap)
    ax1.set_xlabel("post-synaptic neuronal idx", fontsize=fontsize, labelpad=axis_labelpad)
    ax1.set_ylabel("pre-synaptic neuronal idx", fontsize=fontsize, labelpad=axis_labelpad)
    ax1.set_xticks(axis_ticks)
    ax1.set_xticklabels([str(i) for i in axis_ticks], fontsize=fontsize)
    ax1.set_yticks(axis_ticks)
    ax1.set_yticklabels([str(i) for i in axis_ticks], fontsize=fontsize)
    fig1.savefig(os.path.join(out_path, '2_f_worm_8.pdf'), format='pdf')
    plt.show()

    fig2 = plt.figure(2, figsize=fig_size)
    ax2 = fig2.add_axes(main_axes)
    x0, x1 = ax2.get_xlim()
    y0, y1 = ax2.get_ylim()
    ax2.set_aspect(abs(x1 - x0) / abs(y1 - y0), 'box')
    im2 = ax2.imshow(model_single_draw, cmap=model_cmap)
    ax2.set_xlabel("post-synaptic neuronal idx", fontsize=fontsize, labelpad=axis_labelpad)
    ax2.set_ylabel("pre-synaptic neuronal idx", fontsize=fontsize, labelpad=axis_labelpad)
    ax2.set_xticks(axis_ticks)
    ax2.set_xticklabels([str(i) for i in axis_ticks], fontsize=fontsize)
    ax2.set_yticks(axis_ticks)
    ax2.set_yticklabels([str(i) for i in axis_ticks], fontsize=fontsize)
    fig2.savefig(os.path.join(out_path, '2_g.pdf'), format='pdf')
    plt.show()


def fig_graph_features_a(out_path="Figures\\FigGraphFeatures"):
    data_path = "CElegansData\SubTypes\\connectomes\Dataset8.pkl"
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    data_mat = nx.to_numpy_array(data, nodelist=sorted(data.nodes))
    data_in_degrees = data_mat.sum(axis=0)
    data_out_degrees = data_mat.sum(axis=1)

    num_types = 8
    model_average_mat_path = f"SavedOutputs\IndependentModel\\average_adj_mats\InferredTypes\\{num_types}_types.pkl"
    with open(model_average_mat_path, 'rb') as f:
        model_average_mat = pickle.load(f)
    num_neurons = model_average_mat.shape[0]

    model_average_in_deg_hist, model_in_deg_hist_std, model_average_out_deg_hist, model_out_deg_hist_std = average_degree_distribution(
        model_average_mat)

    data_in_deg_hist, _ = np.histogram(data_in_degrees, bins=range(num_neurons + 1))
    max_in_deg = max(np.max(np.argwhere(data_in_deg_hist > 0)),
                     np.max(np.argwhere(model_average_in_deg_hist > 10e-6)))
    data_out_deg_hist, _ = np.histogram(data_out_degrees, bins=range(num_neurons + 1))
    num_stds = 2

    fig = plt.figure(figsize=RECT_SMALL_FIG_SIZE)
    fontsize = FONT_SIZE
    markersize = MARKER_SIZE * 3
    line_width = LINE_WIDTH
    main_axes = [0.23, 0.23, 0.71, 0.76]
    axes_labelpad = 2
    pylab.rcParams['xtick.major.pad'] = '0.5'
    pylab.rcParams['ytick.major.pad'] = '0.5'
    ax1 = fig.add_axes(main_axes)
    ax1.set_xlim(-1, 51)
    ax1.set_ylim(0, 30)
    x_axes_ticks = np.arange(0, 55, 10)
    y_axes_ticks = np.arange(0, 30, 5)
    ax1.set_xticks(x_axes_ticks)
    ax1.set_xticklabels([f'{tick}' for tick in x_axes_ticks], fontsize=fontsize)
    ax1.set_yticks(y_axes_ticks)
    ax1.set_yticklabels([f'{tick}' for tick in y_axes_ticks], fontsize=fontsize)
    ax1.set_xlabel('in-degree', fontsize=fontsize, labelpad=axes_labelpad)
    ax1.set_ylabel('frequency', fontsize=fontsize, labelpad=axes_labelpad)

    ax1.plot(range(max_in_deg + 1), data_in_deg_hist[:max_in_deg + 1], marker='.', markersize=markersize, lw=line_width,
             label="data", color='k')
    ax1.fill_between(range(max_in_deg + 1),
                     y1=model_average_in_deg_hist[:max_in_deg + 1] + num_stds * model_in_deg_hist_std[:max_in_deg + 1],
                     y2=np.clip(
                         model_average_in_deg_hist[:max_in_deg + 1] - num_stds * model_in_deg_hist_std[:max_in_deg + 1],
                         a_min=0, a_max=None), color=SINGLE_EPOCH_INFERRED_COLOR, alpha=0.5, label="model")
    plt.savefig(os.path.join(out_path, 'in_deg_dist.pdf'), format='pdf')
    plt.show()

    max_out_deg = max(np.max(np.argwhere(data_out_deg_hist > 0)),
                      np.max(np.argwhere(model_average_out_deg_hist > 10e-6)))

    fig = plt.figure(figsize=RECT_SMALL_FIG_SIZE)
    ax1 = fig.add_axes(main_axes)
    ax1.set_xlim(-1, 51)
    ax1.set_ylim(0, 21)
    x_axes_ticks = np.arange(0, 55, 10)
    y_axes_ticks = np.arange(0, 25, 5)
    ax1.set_xticks(x_axes_ticks)
    ax1.set_xticklabels([f'{tick}' for tick in x_axes_ticks], fontsize=fontsize)
    ax1.set_yticks(y_axes_ticks)
    ax1.set_yticklabels([f'{tick}' for tick in y_axes_ticks], fontsize=fontsize)
    ax1.set_xlabel('out-degree', fontsize=fontsize, labelpad=axes_labelpad)
    ax1.set_ylabel('frequency', fontsize=fontsize, labelpad=axes_labelpad)

    ax1.plot(range(max_out_deg + 1), data_out_deg_hist[:max_out_deg + 1], marker='.', markersize=markersize,
             lw=line_width, label="data", color='k')
    ax1.fill_between(range(max_out_deg + 1),
                     y1=model_average_out_deg_hist[:max_out_deg + 1] + num_stds * model_out_deg_hist_std[
                                                                                  :max_out_deg + 1],
                     y2=np.clip(model_average_out_deg_hist[:max_out_deg + 1] - num_stds * model_out_deg_hist_std[
                                                                                          :max_out_deg + 1], a_min=0,
                                a_max=None), color=SINGLE_EPOCH_INFERRED_COLOR, alpha=0.5, label="model")
    plt.savefig(os.path.join(out_path, 'out_deg_dist.pdf'), format='pdf')
    plt.show()


def fig_graph_features_b(out_path="Figures\\FigGraphFeatures"):
    num_types = 8
    model_average_mat_path = f"SavedOutputs\IndependentModel\\average_adj_mats\InferredTypes\\{num_types}_types.pkl"
    with open(model_average_mat_path, 'rb') as f:
        model_average_mat = pickle.load(f)
    num_neurons = model_average_mat.shape[0]
    model_mean_in_degrees = model_average_mat.sum(axis=0)
    model_std_in_degrees = np.sqrt(np.sum(model_average_mat * (1 - model_average_mat), axis=0))
    model_mean_out_degrees = model_average_mat.sum(axis=1)
    model_std_out_degrees = np.sqrt(np.sum(model_average_mat * (1 - model_average_mat), axis=1))

    data_path = "CElegansData\SubTypes\\connectomes\Dataset8.pkl"
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    data_mat = nx.to_numpy_array(data, nodelist=sorted(data.nodes))
    data_in_degrees = data_mat.sum(axis=0)
    data_out_degrees = data_mat.sum(axis=1)

    sorted_in_degrees_indices = data_in_degrees.argsort()
    num_stds = 2

    figsize = RECT_SMALL_FIG_SIZE
    fontsize = FONT_SIZE
    markersize = MARKER_SIZE * 3
    line_width = LINE_WIDTH
    main_axes = [0.23, 0.23, 0.71, 0.76]
    axes_labelpad = 2
    pylab.rcParams['xtick.major.pad'] = '0.5'
    pylab.rcParams['ytick.major.pad'] = '0.5'

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(main_axes)
    ax.plot(range(1, num_neurons + 1), data_in_degrees[sorted_in_degrees_indices], '.',
            markersize=markersize, lw=line_width, label='data', color='k')
    ax.fill_between(range(1, num_neurons + 1),
                    y1=model_mean_in_degrees[sorted_in_degrees_indices] + num_stds * model_std_in_degrees[
                        sorted_in_degrees_indices],
                    y2=model_mean_in_degrees[sorted_in_degrees_indices] - num_stds * model_std_in_degrees[
                        sorted_in_degrees_indices], color=SINGLE_EPOCH_INFERRED_COLOR, alpha=0.5, label='model')
    ax.set_xlim(-5, 190)
    ax.set_ylim(-2, 47)
    x_axes_ticks = np.arange(0, 181, 60)
    y_axes_ticks = np.arange(0, 46, 15)
    ax.set_xticks(x_axes_ticks)
    ax.set_xticklabels([f'{tick}' for tick in x_axes_ticks], fontsize=fontsize)
    ax.set_yticks(y_axes_ticks)
    ax.set_yticklabels([f'{tick}' for tick in y_axes_ticks], fontsize=fontsize)
    ax.set_xlabel('neuronal index', fontsize=fontsize, labelpad=axes_labelpad)
    ax.set_ylabel('in-degree', fontsize=fontsize, labelpad=axes_labelpad)
    plt.savefig(os.path.join(out_path, 'individual_in_degs.pdf'), format='pdf')
    plt.show()

    sorted_out_degrees_indices = data_out_degrees.argsort()
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(main_axes)
    ax.plot(range(1, num_neurons + 1), data_out_degrees[sorted_out_degrees_indices], '.',
            markersize=markersize, lw=line_width, label='data', color='k')
    ax.fill_between(range(1, num_neurons + 1),
                    y1=model_mean_out_degrees[sorted_out_degrees_indices] + num_stds * model_std_out_degrees[
                        sorted_out_degrees_indices],
                    y2=model_mean_out_degrees[sorted_out_degrees_indices] - num_stds * model_std_out_degrees[
                        sorted_out_degrees_indices], color=SINGLE_EPOCH_INFERRED_COLOR, alpha=0.5, label='model')
    ax.set_xlim(-5, 190)
    ax.set_ylim(-2, 47)
    x_axes_ticks = np.arange(0, 181, 60)
    y_axes_ticks = np.arange(0, 46, 15)
    ax.set_xticks(x_axes_ticks)
    ax.set_xticklabels([f'{tick}' for tick in x_axes_ticks], fontsize=fontsize)
    ax.set_yticks(y_axes_ticks)
    ax.set_yticklabels([f'{tick}' for tick in y_axes_ticks], fontsize=fontsize)
    ax.set_xlabel('neuronal index', fontsize=fontsize, labelpad=axes_labelpad)
    ax.set_ylabel('out-degree', fontsize=fontsize, labelpad=axes_labelpad)
    plt.savefig(os.path.join(out_path, 'individual_out_degs.pdf'), format='pdf')
    plt.show()


def fig_graph_features_c_d(out_path="Figures\\FigGraphFeatures"):
    birth_times_res = 10  # min.
    data_path = "CElegansData\SubTypes\\connectomes\Dataset8.pkl"
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    sorted_neurons = sorted(data.nodes)
    num_neurons = len(sorted_neurons)
    syn_lengths = np.zeros((num_neurons, num_neurons))
    syn_birth_times = np.zeros((num_neurons, num_neurons))
    for i in range(num_neurons):
        for j in range(num_neurons):
            if i == j:
                continue
            cur_pre_neuron = sorted_neurons[i]
            cur_post_neuron = sorted_neurons[j]
            cur_syn_len = np.sqrt(
                np.sum((data.nodes[cur_pre_neuron]['coords'] * WORM_LENGTH_NORMALIZATION - data.nodes[cur_post_neuron][
                    'coords'] * WORM_LENGTH_NORMALIZATION) ** 2))
            syn_lengths[i, j] = cur_syn_len
            syn_birth_times[i, j] = max(data.nodes[cur_pre_neuron]['birth_time'],
                                        data.nodes[cur_post_neuron]['birth_time'])
    max_syn_length = np.max(syn_lengths)
    len_resolution = 0.1
    len_bins = np.arange(0, max_syn_length + len_resolution, len_resolution)
    data_synaptic_lengths = [data.get_edge_data(syn[0], syn[1])['length'] * WORM_LENGTH_NORMALIZATION for syn in
                             data.edges]
    data_synaptic_lengths_hist, _ = np.histogram(data_synaptic_lengths, bins=len_bins)

    birth_times_resolution = birth_times_res
    birth_times_bins = np.arange(0, ADULT_WORM_AGE + birth_times_resolution, birth_times_resolution)
    data_synaptic_birth_times = [data.get_edge_data(syn[0], syn[1])['birth time'] for syn in data.edges]
    data_synaptic_birth_times_hist, _ = np.histogram(data_synaptic_birth_times, bins=birth_times_bins)

    num_types = 8
    model_average_mat_path = f"SavedOutputs\IndependentModel\\average_adj_mats\InferredTypes\\{num_types}_types.pkl"
    with open(model_average_mat_path, 'rb') as f:
        model_average_mat = pickle.load(f)
    mean_model_syn_len_cumulative_hist = np.zeros(len_bins.size - 1)
    std_model_syn_len_cumulative_hist = np.zeros(len_bins.size - 1)
    mean_model_syn_birth_times_cumulative_hist = np.zeros(birth_times_bins.size - 1)
    std_model_syn_birth_times_cumulative_hist = np.zeros(birth_times_bins.size - 1)
    for i in range(model_average_mat.shape[0]):
        for j in range(model_average_mat.shape[1]):
            if i == j:
                continue
            cur_syn_prob = model_average_mat[i, j]
            cur_syn_len = syn_lengths[i, j]
            cur_syn_birth_time = syn_birth_times[i, j]
            for bin_idx in range(len_bins.size - 1):
                if cur_syn_len <= len_bins[bin_idx + 1]:
                    mean_model_syn_len_cumulative_hist[bin_idx] += cur_syn_prob
                    std_model_syn_len_cumulative_hist[bin_idx] += cur_syn_prob * (1 - cur_syn_prob)
            for bin_idx in range(birth_times_bins.size - 1):
                if cur_syn_birth_time <= birth_times_bins[bin_idx + 1]:
                    mean_model_syn_birth_times_cumulative_hist[bin_idx] += cur_syn_prob
                    std_model_syn_birth_times_cumulative_hist[bin_idx] += cur_syn_prob * (1 - cur_syn_prob)
    std_model_syn_len_cumulative_hist = np.sqrt(std_model_syn_len_cumulative_hist)
    std_model_syn_birth_times_cumulative_hist = np.sqrt(std_model_syn_birth_times_cumulative_hist)

    num_stds = 2
    fontsize = FONT_SIZE
    line_width = LINE_WIDTH
    main_axes = [0.23, 0.23, 0.71, 0.76]
    axes_labelpad = 1
    y_label_coords = (-0.24, 0.46)
    pylab.rcParams['xtick.major.pad'] = '0.5'
    pylab.rcParams['ytick.major.pad'] = '0.5'
    fig = plt.figure(figsize=RECT_SMALL_FIG_SIZE)
    ax1 = fig.add_axes(main_axes)
    ax1.set_xlim(-50, 850)
    ax1.set_ylim(0, 2100)
    x_axes_ticks = np.arange(0, 900, 200)
    y_axes_ticks = np.arange(0, 2050, 500)
    ax1.set_xticks(x_axes_ticks)
    ax1.set_xticklabels([f'{tick}' if list(x_axes_ticks).index(tick) % 2 == 0 else '' for tick in x_axes_ticks],
                        fontsize=fontsize)
    ax1.set_yticks(y_axes_ticks)
    ax1.set_yticklabels([f'{tick}' if list(y_axes_ticks).index(tick) % 2 == 0 else '' for tick in y_axes_ticks],
                        fontsize=fontsize)
    ax1.set_xlabel('distance between neurons [$\mu m$]', fontsize=fontsize, labelpad=axes_labelpad)
    ax1.set_ylabel('cumulative frequency', fontsize=fontsize, labelpad=axes_labelpad)
    ax1.yaxis.set_label_coords(y_label_coords[0], y_label_coords[1])

    ax1.plot(len_bins[1:], np.cumsum(data_synaptic_lengths_hist), lw=line_width, label='data', color='k')
    ax1.fill_between(len_bins[1:], y1=mean_model_syn_len_cumulative_hist + num_stds * std_model_syn_len_cumulative_hist,
                     y2=mean_model_syn_len_cumulative_hist - num_stds * std_model_syn_len_cumulative_hist,
                     color=SINGLE_EPOCH_INFERRED_COLOR,
                     alpha=0.5, label='model')
    plt.savefig(os.path.join(out_path, 'connected_neurons_dist.pdf'), format='pdf')
    plt.show()

    fig = plt.figure(figsize=RECT_SMALL_FIG_SIZE)
    ax1 = fig.add_axes(main_axes)
    ax1.set_xlim(-50, ADULT_WORM_AGE + 50)
    ax1.set_ylim(0, 2100)
    x_axes_ticks = np.arange(0, ADULT_WORM_AGE + 50, 700)
    y_axes_ticks = np.arange(0, 2050, 500)
    ax1.set_xticks(x_axes_ticks)
    ax1.set_xticklabels([f'{tick}' if list(x_axes_ticks).index(tick) % 2 == 0 else '' for tick in x_axes_ticks],
                        fontsize=fontsize)
    ax1.set_yticks(y_axes_ticks)
    ax1.set_yticklabels([f'{tick}' if list(y_axes_ticks).index(tick) % 2 == 0 else '' for tick in y_axes_ticks],
                        fontsize=fontsize)
    ax1.set_xlabel('worm age [min.]', fontsize=fontsize, labelpad=axes_labelpad)
    ax1.set_ylabel('cumulative frequency', fontsize=fontsize, labelpad=axes_labelpad)
    ax1.yaxis.set_label_coords(y_label_coords[0], y_label_coords[1])
    plt.plot(birth_times_bins[1:], np.cumsum(data_synaptic_birth_times_hist), lw=line_width, label='data', color='k')
    plt.fill_between(birth_times_bins[1:],
                     y1=mean_model_syn_birth_times_cumulative_hist + num_stds * std_model_syn_birth_times_cumulative_hist,
                     y2=mean_model_syn_birth_times_cumulative_hist - num_stds * std_model_syn_birth_times_cumulative_hist,
                     color=SINGLE_EPOCH_INFERRED_COLOR, alpha=0.5, label='model')
    plt.savefig(os.path.join(out_path, 'connected_neurons_birth_times.pdf'), format='pdf')
    plt.show()


def fig_graph_features_e(out_path="Figures\\FigGraphFeatures",
                         saved_calcs_path="Figures\SavedCalcs\\triads_distributions\independent_model",
                         is_saved=False, do_save=True):
    data_path = "CElegansData\SubTypes\\connectomes\Dataset8.pkl"
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    data_triads_distribution = calc_norm_triad_motifs_dist(data)

    num_types = 8
    model_outputs_path = f"SavedOutputs\IndependentModel\connectomes\InferredTypes\\{num_types}_types"
    files = os.listdir(model_outputs_path)
    model_triad_distributions = np.zeros((NUM_MODEL_SAMPLES_FOR_STATISTICS, NUM_TRIAD_TYPES))
    for i in range(NUM_MODEL_SAMPLES_FOR_STATISTICS):
        if not is_saved:
            with open(os.path.join(model_outputs_path, files[i]), 'rb') as f:
                model_sample_graph = pickle.load(f)
            model_triad_distributions[i] = calc_norm_triad_motifs_dist(model_sample_graph)
            if do_save:
                with open(os.path.join(saved_calcs_path, f'{i}.pkl'), 'wb') as f:
                    pickle.dump(model_triad_distributions[i], f)
        else:
            with open(os.path.join(saved_calcs_path, f'{i}.pkl'), 'rb') as f:
                model_triad_distributions[i] = pickle.load(f)
    model_triads_distributions_mean = model_triad_distributions.mean(axis=0)
    model_triad_distributions_std = model_triad_distributions.std(axis=0)

    fig = plt.figure(figsize=(8.4 * CM_TO_INCH, 5.467 * CM_TO_INCH))
    fontsize = FONT_SIZE
    pylab.rcParams['xtick.major.pad'] = '0.5'
    pylab.rcParams['ytick.major.pad'] = '0.5'
    axes_labelpad = 1
    main_axes = [0.15, 0.15, 0.84, 0.8]

    ax1 = fig.add_axes(main_axes)
    plt.yscale("log")
    num_stds = 2
    ax1.set_xlim(-0.5, 15.5)
    ax1.set_ylim(10e-6, 1)
    x_axes_ticks = np.arange(0, 16, 1)
    y_axes_ticks = 10.0 ** np.arange(-5, 1, 1)
    ax1.set_xticks(x_axes_ticks)
    ax1.set_xticklabels(['' for i in SORTED_TRIAD_TYPES], rotation=270, fontsize=fontsize)
    ax1.set_yticks(y_axes_ticks)
    ax1.set_yticklabels(['$10^{-5}$', '$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$10^{0}$'],
                        fontsize=fontsize)
    ax1.set_ylabel('triad fraction', fontsize=fontsize, labelpad=axes_labelpad)
    width = 0.3
    ax1.bar(np.arange(len(SORTED_TRIAD_TYPES)) - width / 2, data_triads_distribution, align='center', width=width,
            label="data", color='k')
    ax1.bar(np.arange(len(SORTED_TRIAD_TYPES)) + width / 2, model_triads_distributions_mean,
            yerr=num_stds * model_triad_distributions_std,
            color=SINGLE_EPOCH_INFERRED_COLOR, ecolor=SINGLE_EPOCH_INFERRED_COLOR, align='center', width=width,
            label="model")
    plt.savefig(os.path.join(out_path, 'triads_no_reciprocity.pdf'), format='pdf')
    plt.show()


def fig_graph_features_f(out_path="Figures\\FigGraphFeatures",
                         saved_calcs_path="Figures\SavedCalcs\\triads_distributions\\reciprocal_model",
                         is_saved=False, do_save=True):
    data_path = "CElegansData\SubTypes\\connectomes\\Dataset8.pkl"
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    data_triads = calc_norm_triad_motifs_dist(data)

    gamma = 7
    num_types = 8
    connectomes_path = f"SavedOutputs\ReciprocalModel\FullDataset\connectomes\\{num_types}_types\gamma{gamma}"
    files_list = os.listdir(connectomes_path)
    num_files = len(files_list)
    model_triads = np.zeros((num_files, NUM_TRIAD_TYPES))
    idx = 0
    for file_name in files_list:
        if not is_saved:
            with open(os.path.join(connectomes_path, file_name), 'rb') as f:
                connectome = pickle.load(f)
            model_triads[idx] = calc_norm_triad_motifs_dist(connectome)
            if do_save:
                with open(os.path.join(saved_calcs_path, file_name), 'wb') as f:
                    pickle.dump(model_triads[idx], f)
        else:
            with open(os.path.join(saved_calcs_path, file_name), 'rb') as f:
                model_triads[idx] = pickle.load(f)
        idx += 1

    fig = plt.figure(figsize=(8.4 * CM_TO_INCH, 5.467 * CM_TO_INCH))
    fontsize = FONT_SIZE
    pylab.rcParams['xtick.major.pad'] = '0.5'
    pylab.rcParams['ytick.major.pad'] = '0.5'
    axes_labelpad = 1
    main_axes = [0.15, 0.15, 0.84, 0.8]
    ax1 = fig.add_axes(main_axes)
    plt.yscale("log")
    num_stds = 2
    ax1.set_xlim(-0.5, 15.5)
    ax1.set_ylim(10e-6, 1)
    x_axes_ticks = np.arange(0, 16, 1)
    y_axes_ticks = 10.0 ** np.arange(-5, 1, 1)
    ax1.set_xticks(x_axes_ticks)
    ax1.set_xticklabels(['' for i in SORTED_TRIAD_TYPES], rotation=270, fontsize=fontsize)
    ax1.set_yticks(y_axes_ticks)
    ax1.set_yticklabels(['$10^{-5}$', '$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$10^{0}$'],
                        fontsize=fontsize)
    ax1.set_ylabel('triad fraction', fontsize=fontsize, labelpad=axes_labelpad)
    width = 0.3
    ax1.bar(np.arange(len(SORTED_TRIAD_TYPES)) - width / 2, data_triads, align='center', width=width,
            label="data", color='k')
    ax1.bar(np.arange(len(SORTED_TRIAD_TYPES)) + width / 2, model_triads.mean(axis=0),
            yerr=num_stds * model_triads.std(axis=0),
            color=SINGLE_EPOCH_INFERRED_COLOR, ecolor=SINGLE_EPOCH_INFERRED_COLOR, align='center', width=width,
            label="model")
    plt.savefig(os.path.join(out_path, 'traids_with_reciprocity.pdf'), format='pdf')
    plt.show()


def fig_inferred_types_bio_interpretation_a(out_path="Figures\\FigInfTypesBioInter"):
    num_inferred_types = 8
    art_types_worm_7_by_names = convert_indices_to_names_in_artificial_types(
        f'CElegansData\InferredTypes\\types\\{num_inferred_types}.pkl',
        'CElegansData\\nerve_ring_neurons_subset.pkl')
    cook_types_list, cook_types_names = create_cook_types_list('CElegansData\\nerve_ring_neurons_subset.pkl')
    with open('CElegansData\\nerve_ring_neurons_subset.pkl', 'rb') as f:
        neurons_list = pickle.load(f)

    cook_inferred_join_prob = generate_cluster_intersection_joint_prob_map(neurons_list, cook_types_list,
                                                                           art_types_worm_7_by_names)

    sorted_cook_types_names = sorted(cook_types_names)
    cook_sorted_indices = []
    for i in range(len(cook_types_names)):
        cook_sorted_indices.append(cook_types_names.index(sorted_cook_types_names[i]))
    sorted_cook_types_names_copy = sorted_cook_types_names.copy()
    sorted_cook_types_names[:5] = sorted_cook_types_names_copy[-5:]
    sorted_cook_types_names[-5:] = sorted_cook_types_names_copy[:5]
    cook_sorted_indices_copy = cook_sorted_indices.copy()
    cook_sorted_indices[:5] = cook_sorted_indices_copy[-5:]
    cook_sorted_indices[-5:] = cook_sorted_indices_copy[:5]
    cook_sorted_indices = np.array(cook_sorted_indices).reshape(len(cook_types_names), 1)
    inferred_sorted_indices = []
    for i in range(num_inferred_types):
        inferred_sorted_indices.append(INFERRED_TYPES_LABEL_TO_INDEX[f'C{i}'])
    inferred_sorted_indices = np.array(inferred_sorted_indices).reshape(1, num_inferred_types)
    cook_inferred_join_prob = cook_inferred_join_prob[cook_sorted_indices, inferred_sorted_indices]

    max_value = cook_inferred_join_prob.max()
    min_value = 0
    fontsize = FONT_SIZE
    height = 0.82
    width = num_inferred_types / 13 * height
    x_start = 0.17
    y_start = 0.17
    main_axes = [x_start, y_start, width, height]
    colorbar_axes = [x_start + width + 0.05, y_start, 0.05, height]
    pylab.rcParams['xtick.major.pad'] = '0.5'
    pylab.rcParams['ytick.major.pad'] = '0.5'
    axis_labelpad = 1
    colorbar_labelpad = 10
    cook_types_color = COOK_TYPES_COLOR
    inferred_types_color = 'darkturquoise'
    fig = plt.figure(figsize=SQUARE_FIG_SIZE)
    ax = fig.add_axes(main_axes)
    im = ax.imshow(cook_inferred_join_prob,
                   cmap=colors.LinearSegmentedColormap.from_list('steelblue', ['white', 'steelblue']), vmin=min_value,
                   vmax=max_value)
    cbar_ax = fig.add_axes(colorbar_axes)
    cbar_ticks = np.arange(0, 0.11, 0.025)
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=cbar_ticks)
    cbar_ax.set_yticklabels(
        [f'{cbar_ticks[0]:.2f}', '', f'{cbar_ticks[2]:.2f}', '', f'{cbar_ticks[4]:.2f}'],
        fontsize=fontsize)
    cbar.ax.set_ylabel('overlap', rotation=270, fontsize=fontsize, labelpad=colorbar_labelpad)
    ax.set_ylabel('traditional type', fontsize=fontsize, labelpad=axis_labelpad, color=cook_types_color)
    ax.set_xlabel('inferred type', fontsize=fontsize, labelpad=axis_labelpad, color=inferred_types_color)
    ax.set_xticks(np.arange(num_inferred_types))
    ax.set_xticklabels([f'C{i}' for i in range(num_inferred_types)], fontsize=fontsize, rotation=315,
                       color=inferred_types_color)
    ax.set_yticks(np.arange(len(cook_types_names)))
    ax.set_yticklabels(sorted_cook_types_names, fontsize=fontsize, color=cook_types_color)
    plt.savefig(os.path.join(out_path, f'panel_a_{num_inferred_types}_types.pdf'), format='pdf')
    plt.show()


def fig_inferred_types_bio_interpretation_b(out_path="Figures\\FigInfTypesBioInter"):
    num_inferred_types = 8
    num_cook_types = 13
    data_path = f"CElegansData\InferredTypes\\connectomes\\{num_inferred_types}_types\\Dataset7.pkl"
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    with open("CElegansData\\neuronal_types_dict.pkl", 'rb') as f:
        neuronal_cook_types = pickle.load(f)

    lengths_inferred_types = {}
    birth_times_inferred_types = {}
    lengths_cook_types = {}
    birth_times_cook_types = {}
    for pre in data.nodes:
        for post in data.nodes:
            if pre == post:
                continue
            cur_len = np.sqrt(
                (((data.nodes[pre]['coords'] - data.nodes[post]['coords']) * WORM_LENGTH_NORMALIZATION) ** 2).sum())
            cur_birth_time = max(data.nodes[pre]['birth_time'], data.nodes[post]['birth_time'])
            cur_inferred_type = data.nodes[pre]['type'] * num_inferred_types + data.nodes[post]['type']
            cur_cook_type = (neuronal_cook_types[pre][CElegansNeuronsAdder.SUB_TYPES],
                             neuronal_cook_types[post][CElegansNeuronsAdder.SUB_TYPES])
            if cur_inferred_type in lengths_inferred_types.keys():
                lengths_inferred_types[cur_inferred_type].append(cur_len)
                birth_times_inferred_types[cur_inferred_type].append(cur_birth_time)
            else:
                lengths_inferred_types[cur_inferred_type] = [cur_len]
                birth_times_inferred_types[cur_inferred_type] = [cur_birth_time]
            if cur_cook_type in lengths_cook_types.keys():
                lengths_cook_types[cur_cook_type].append(cur_len)
                birth_times_cook_types[cur_cook_type].append(cur_birth_time)
            else:
                lengths_cook_types[cur_cook_type] = [cur_len]
                birth_times_cook_types[cur_cook_type] = [cur_birth_time]

    average_lengths_inferred = np.zeros(num_inferred_types ** 2)
    std_lengths_inferred = np.zeros(num_inferred_types ** 2)
    sem_lengths_inferred = np.zeros(num_inferred_types ** 2)
    average_birth_times_inferred = np.zeros(num_inferred_types ** 2)
    std_birth_times_inferred = np.zeros(num_inferred_types ** 2)
    sem_birth_times_inferred = np.zeros(num_inferred_types ** 2)
    idx = 0
    for t in lengths_inferred_types.keys():
        average_lengths_inferred[idx] = np.array(lengths_inferred_types[t]).mean()
        std_lengths_inferred[idx] = np.array(lengths_inferred_types[t]).std()
        sem_lengths_inferred[idx] = std_lengths_inferred[idx] / np.sqrt(len(lengths_inferred_types[t]))
        average_birth_times_inferred[idx] = np.array(birth_times_inferred_types[t]).mean()
        std_birth_times_inferred[idx] = np.array(birth_times_inferred_types[t]).std()
        sem_birth_times_inferred[idx] = std_birth_times_inferred[idx] / np.sqrt(len(birth_times_inferred_types[t]))
        idx += 1

    average_lengths_cook = np.zeros(num_cook_types ** 2)
    std_lengths_cook = np.zeros(num_cook_types ** 2)
    sem_lengths_cook = np.zeros(num_cook_types ** 2)
    average_birth_times_cook = np.zeros(num_cook_types ** 2)
    std_birth_times_cook = np.zeros(num_cook_types ** 2)
    sem_birth_times_cook = np.zeros(num_cook_types ** 2)
    idx = 0
    for t in lengths_cook_types.keys():
        average_lengths_cook[idx] = np.array(lengths_cook_types[t]).mean()
        std_lengths_cook[idx] = np.array(lengths_cook_types[t]).std()
        sem_lengths_cook[idx] = std_lengths_cook[idx] / np.sqrt(len(lengths_cook_types[t]))
        average_birth_times_cook[idx] = np.array(birth_times_cook_types[t]).mean()
        std_birth_times_cook[idx] = np.array(birth_times_cook_types[t]).std()
        sem_birth_times_cook[idx] = std_birth_times_cook[idx] / np.sqrt(len(birth_times_cook_types[t]))
        idx += 1

    sorted_indices_average_lengths_inferred = np.argsort(average_lengths_inferred)
    sorted_indices_average_birth_times_inferred = np.argsort(average_birth_times_inferred)
    sorted_indices_average_lengths_cook = np.argsort(average_lengths_cook)
    sorted_indices_average_birth_times_cook = np.argsort(average_birth_times_cook)

    fig_size = SQUARE_FIG_SIZE
    fontsize = FONT_SIZE
    pylab.rcParams['xtick.major.pad'] = '0.5'
    pylab.rcParams['ytick.major.pad'] = '0.5'
    axis_labelpad = 1
    main_axes = [0.17, 0.18, 0.81, 0.8]
    cook_types_color = COOK_TYPES_COLOR
    inferred_types_color = 'darkturquoise'

    fig = plt.figure(figsize=fig_size)
    ax1 = fig.add_axes(main_axes)
    x_axes_ticks = np.arange(0, 151, 50)
    y_axes_ticks = np.arange(0, 451, 150)
    ax1.set_xticks(x_axes_ticks)
    ax1.set_xticklabels([f'{tick}' for tick in x_axes_ticks], fontsize=fontsize)
    ax1.set_yticks(y_axes_ticks)
    ax1.set_yticklabels(
        [f'{y_axes_ticks[i]}' if i == 0 or i == len(y_axes_ticks) - 1 else '' for i in range(len(y_axes_ticks))],
        fontsize=fontsize)
    ax1.set_xlabel('synaptic type index', fontsize=fontsize, labelpad=axis_labelpad)
    ax1.set_ylabel('distance between\nneurons [$\mu m$]', fontsize=fontsize, labelpad=axis_labelpad - 15)
    ax1.set_xlim(-2, num_cook_types ** 2 + 2)
    ax1.set_ylim(0, max(average_lengths_inferred.max() + sem_lengths_inferred.max(),
                        average_lengths_cook.max() + sem_lengths_cook.max()))

    ax1.plot(range(num_cook_types ** 2), average_lengths_cook[sorted_indices_average_lengths_cook],
             color=cook_types_color)
    ax1.fill_between(range(num_cook_types ** 2),
                     average_lengths_cook[sorted_indices_average_lengths_cook] - sem_lengths_cook[
                         sorted_indices_average_lengths_cook],
                     average_lengths_cook[sorted_indices_average_lengths_cook] + sem_lengths_cook[
                         sorted_indices_average_lengths_cook],
                     alpha=0.5, color=cook_types_color)

    ax1.plot(range(num_inferred_types ** 2), average_lengths_inferred[sorted_indices_average_lengths_inferred],
             color=inferred_types_color)
    ax1.fill_between(range(num_inferred_types ** 2),
                     average_lengths_inferred[sorted_indices_average_lengths_inferred] - sem_lengths_inferred[
                         sorted_indices_average_lengths_inferred],
                     average_lengths_inferred[sorted_indices_average_lengths_inferred] + sem_lengths_inferred[
                         sorted_indices_average_lengths_inferred],
                     alpha=0.5, color=inferred_types_color)
    plt.savefig(os.path.join(out_path, f"lengths_{num_inferred_types}types.pdf"), format='pdf')
    plt.show()

    fig = plt.figure(figsize=fig_size)
    ax1 = fig.add_axes(main_axes)
    x_axes_ticks = np.arange(0, 151, 50)
    y_axes_ticks = np.arange(300, 1201, 300)
    ax1.set_xticks(x_axes_ticks)
    ax1.set_xticklabels([f'{tick}' for tick in x_axes_ticks], fontsize=fontsize)
    ax1.set_yticks(y_axes_ticks)
    ax1.set_yticklabels(
        [f'{y_axes_ticks[i]}' if i == 0 or i == len(y_axes_ticks) - 1 else '' for i in range(len(y_axes_ticks))],
        fontsize=fontsize)
    ax1.set_xlabel('synaptic type index', fontsize=fontsize, labelpad=axis_labelpad)
    ax1.set_ylabel('neuronal maximal\nbirth time [min.]', fontsize=fontsize, labelpad=axis_labelpad - 17)
    ax1.set_xlim(-2, num_cook_types ** 2 + 2)
    ax1.set_ylim(200, max(average_birth_times_inferred.max() + sem_birth_times_inferred.max(),
                          average_birth_times_cook.max() + sem_birth_times_cook.max()))

    plt.plot(range(num_cook_types ** 2), average_birth_times_cook[sorted_indices_average_birth_times_cook],
             color=cook_types_color)
    plt.fill_between(range(num_cook_types ** 2),
                     average_birth_times_cook[sorted_indices_average_birth_times_cook] -
                     sem_birth_times_cook[
                         sorted_indices_average_birth_times_cook],
                     average_birth_times_cook[sorted_indices_average_birth_times_cook] +
                     sem_birth_times_cook[
                         sorted_indices_average_birth_times_cook],
                     alpha=0.5, color=cook_types_color)

    plt.plot(range(num_inferred_types ** 2), average_birth_times_inferred[sorted_indices_average_birth_times_inferred],
             color=inferred_types_color)
    plt.fill_between(range(num_inferred_types ** 2),
                     average_birth_times_inferred[sorted_indices_average_birth_times_inferred] -
                     sem_birth_times_inferred[
                         sorted_indices_average_birth_times_inferred],
                     average_birth_times_inferred[sorted_indices_average_birth_times_inferred] +
                     sem_birth_times_inferred[
                         sorted_indices_average_birth_times_inferred],
                     alpha=0.5, color=inferred_types_color)
    plt.savefig(os.path.join(out_path, f"birth_times_{num_inferred_types}types.pdf"), format='pdf')
    plt.show()


def fig_inferred_types_bio_interpretation_c_d(out_path="Figures\\FigInfTypesBioInter"):
    neuronal_list_path = "CElegansData\\nerve_ring_neurons_subset.pkl"
    cook_types, _ = create_cook_types_list(neuronal_list_path)
    functional_neuronal_types = convert_names_to_indices_neuronal_types(cook_types, neuronal_list_path)
    num_types = 8
    types_path = f"CElegansData\InferredTypes\\types\\{num_types}.pkl"
    with open(types_path, 'rb') as f:
        inferred_neuronal_types = pickle.load(f)[1]

    data_path = f"CElegansData\InferredTypes\\connectomes\\{num_types}_types\\Dataset8.pkl"
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    data_mat = nx.to_numpy_array(data, nodelist=sorted(data.nodes))

    num_inferred_types = len(inferred_neuronal_types)
    inferred_types_mean_in_degs = np.zeros(num_inferred_types)
    inferred_types_std_in_degs = np.zeros(num_inferred_types)
    inferred_types_mean_out_degs = np.zeros(num_inferred_types)
    inferred_types_std_out_degs = np.zeros(num_inferred_types)
    inferred_types_sizes = np.zeros(num_inferred_types)
    type_idx = 0
    for n_type in inferred_neuronal_types:
        num_neurons_in_inferred_type = len(n_type)
        inferred_types_sizes[type_idx] = num_neurons_in_inferred_type
        in_degs = np.zeros(num_neurons_in_inferred_type)
        out_degs = np.zeros(num_neurons_in_inferred_type)
        idx_in_cur_type = 0
        for neuron in n_type:
            in_degs[idx_in_cur_type] = data_mat[:, neuron].sum()
            out_degs[idx_in_cur_type] = data_mat[neuron, :].sum()
            idx_in_cur_type += 1
        inferred_types_mean_in_degs[type_idx] = in_degs.mean()
        inferred_types_std_in_degs[type_idx] = in_degs.std()
        inferred_types_mean_out_degs[type_idx] = out_degs.mean()
        inferred_types_std_out_degs[type_idx] = out_degs.std()
        type_idx += 1

    num_functional_types = len(functional_neuronal_types)
    functional_types_mean_in_degs = np.zeros(num_functional_types)
    functional_types_std_in_degs = np.zeros(num_functional_types)
    functional_types_mean_out_degs = np.zeros(num_functional_types)
    functional_types_std_out_degs = np.zeros(num_functional_types)
    functional_types_sizes = np.zeros(num_functional_types)
    type_idx = 0
    for n_type in functional_neuronal_types:
        num_neurons_in_functional_type = len(n_type)
        functional_types_sizes[type_idx] = num_neurons_in_functional_type
        in_degs = np.zeros(num_neurons_in_functional_type)
        out_degs = np.zeros(num_neurons_in_functional_type)
        idx_in_cur_type = 0
        for neuron in n_type:
            in_degs[idx_in_cur_type] = data_mat[:, neuron].sum()
            out_degs[idx_in_cur_type] = data_mat[neuron, :].sum()
            idx_in_cur_type += 1
        functional_types_mean_in_degs[type_idx] = in_degs.mean()
        functional_types_std_in_degs[type_idx] = in_degs.std()
        functional_types_mean_out_degs[type_idx] = out_degs.mean()
        functional_types_std_out_degs[type_idx] = out_degs.std()
        type_idx += 1

    inferred_in_degs_order = inferred_types_mean_in_degs.argsort()
    inferred_out_degs_order = inferred_types_mean_out_degs.argsort()
    hubbiness_indices = np.zeros(num_inferred_types)
    for i in range(num_inferred_types):
        place_of_i_in_order = np.where(inferred_in_degs_order == i)[0][0]
        place_of_i_out_order = np.where(inferred_out_degs_order == i)[0][0]
        hubbiness_indices[i] = max(place_of_i_in_order, place_of_i_out_order)

    functional_in_degs_order = functional_types_mean_in_degs.argsort()
    functional_out_degs_order = functional_types_mean_out_degs.argsort()
    hubbiness_indices = np.zeros(num_functional_types)
    for i in range(num_functional_types):
        place_of_i_in_order = np.where(functional_in_degs_order == i)[0][0]
        place_of_i_out_order = np.where(functional_out_degs_order == i)[0][0]
        hubbiness_indices[i] = max(place_of_i_in_order, place_of_i_out_order)

    inferred_max_mean_deg = np.zeros(num_inferred_types)
    for i in range(num_inferred_types):
        inferred_max_mean_deg[i] = max(inferred_types_mean_in_degs[i], inferred_types_mean_out_degs[i])
    inferred_max_mean_deg_order = inferred_max_mean_deg.argsort()

    functional_max_mean_deg = np.zeros(num_functional_types)
    for i in range(num_functional_types):
        functional_max_mean_deg[i] = max(functional_types_mean_in_degs[i], functional_types_mean_out_degs[i])
    functional_max_mean_deg_order = functional_max_mean_deg.argsort()

    line_width = LINE_WIDTH
    markersize = MARKER_SIZE * 3
    fontsize = FONT_SIZE
    main_axes = [0.17, 0.18, 0.81, 0.8]
    axes_labelpad = 1

    fig = plt.figure(figsize=SQUARE_FIG_SIZE)
    ax = fig.add_axes(main_axes)
    ax.errorbar(range(1, num_functional_types + 1), functional_types_mean_in_degs[functional_in_degs_order],
                yerr=functional_types_std_in_degs[functional_in_degs_order],
                marker='.', lw=line_width, markersize=markersize, color=COOK_TYPES_COLOR)
    ax.errorbar(range(1, num_inferred_types + 1), inferred_types_mean_in_degs[inferred_in_degs_order],
                yerr=inferred_types_std_in_degs[inferred_in_degs_order],
                marker='.', lw=line_width, markersize=markersize, color='darkturquoise')
    ax.set_xlabel('ascending in-degree type index', fontsize=fontsize, labelpad=axes_labelpad)
    ax.set_ylabel('average in-degree', fontsize=fontsize, labelpad=axes_labelpad)
    ax.set_xlim(0, 14)
    ax.set_ylim(-1, 42)
    x_axis_ticks = range(1, 14, 4)
    y_axis_ticks = range(0, 45, 10)
    ax.set_xticks(x_axis_ticks)
    ax.set_xticklabels([f'{tick}' for tick in x_axis_ticks], fontsize=fontsize)
    ax.set_yticks(y_axis_ticks)
    ax.set_yticklabels([f'{tick}' for tick in y_axis_ticks], fontsize=fontsize)
    plt.savefig(os.path.join(out_path, 'av_in_deg.pdf'), format='pdf')
    plt.show()

    fig = plt.figure(figsize=SQUARE_FIG_SIZE)
    ax = fig.add_axes(main_axes)
    ax.errorbar(range(1, num_functional_types + 1), functional_types_mean_out_degs[functional_out_degs_order],
                yerr=functional_types_std_out_degs[functional_out_degs_order],
                marker='.', lw=line_width, markersize=markersize, color=COOK_TYPES_COLOR)
    ax.errorbar(range(1, num_inferred_types + 1), inferred_types_mean_out_degs[inferred_out_degs_order],
                yerr=inferred_types_std_out_degs[inferred_out_degs_order],
                marker='.', lw=line_width, markersize=markersize, color='darkturquoise')
    ax.set_xlabel('ascending out-degree type index', fontsize=fontsize, labelpad=axes_labelpad)
    ax.set_ylabel('average out-degree', fontsize=fontsize, labelpad=axes_labelpad)
    ax.set_xlim(0, 14)
    ax.set_ylim(-1, 27)
    x_axis_ticks = range(1, 14, 4)
    y_axis_ticks = range(0, 30, 5)
    ax.set_xticks(x_axis_ticks)
    ax.set_xticklabels([f'{tick}' for tick in x_axis_ticks], fontsize=fontsize)
    ax.set_yticks(y_axis_ticks)
    ax.set_yticklabels([f'{tick}' for tick in y_axis_ticks], fontsize=fontsize)
    plt.savefig(os.path.join(out_path, 'av_out_deg.pdf'), format='pdf')
    plt.show()

    fig = plt.figure(figsize=SQUARE_FIG_SIZE)
    ax = fig.add_axes(main_axes)
    ax.plot(functional_max_mean_deg[functional_max_mean_deg_order],
            functional_types_sizes[functional_max_mean_deg_order], marker='.',
            lw=0, markersize=1.5 * markersize, color=COOK_TYPES_COLOR)
    ax.plot(inferred_max_mean_deg[inferred_max_mean_deg_order], inferred_types_sizes[inferred_max_mean_deg_order],
            marker='.',
            lw=0, markersize=1.5 * markersize, color='darkturquoise')
    ax.set_xlabel('$\max(\langle d_{in} \\rangle, \langle d_{out} \\rangle)$', fontsize=fontsize,
                  labelpad=axes_labelpad)
    ax.set_ylabel('number of neurons', fontsize=fontsize, labelpad=axes_labelpad)
    ax.set_xlim(5, 36)
    ax.set_ylim(-1, 55)
    x_axis_ticks = range(5, 36, 5)
    y_axis_ticks = range(0, 55, 10)
    ax.set_xticks(x_axis_ticks)
    ax.set_xticklabels([f'{tick}' for tick in x_axis_ticks], fontsize=fontsize)
    ax.set_yticks(y_axis_ticks)
    ax.set_yticklabels([f'{tick}' for tick in y_axis_ticks], fontsize=fontsize)
    plt.savefig(os.path.join(out_path, 'size_vs_directional_hubbiness.pdf'), format='pdf')
    plt.show()


def _fig_6_a(worms_indices, split):
    np.random.seed(123456789)
    num_types = 8
    data_dir_path = f"CElegansData\InferredTypes\\connectomes\\{num_types}_types"
    dyads_dist_single_epoch_dir_path = f"SavedOutputs\ReciprocalModel\\DyadsSplit\\dyads_distributions\SingleDevStage"
    dyads_dist_multiple_epochs_dir_path = f"SavedOutputs\ReciprocalModel\\DyadsSplit\\dyads_distributions\ThreeDevStages"
    neuronal_types_path = f"CElegansData\InferredTypes\\types\\{num_types}.pkl"
    neurons_list_path = "CElegansData\\nerve_ring_neurons_subset.pkl"

    with open("SavedOutputs\ReciprocalModel\\DyadsSplit\\max_likelihood_params_per_split_single_epoch.pkl",
              'rb') as f:
        smi_single = pickle.load(f)[f'split{split}']['S-']
    with open("SavedOutputs\ReciprocalModel\\DyadsSplit\\max_likelihood_params_per_split_3_epochs.pkl",
              'rb') as f:
        smi_mulitple = pickle.load(f)[f'split{split}']['S-']

    neuronal_types_by_names = convert_indices_to_names_in_artificial_types(neuronal_types_path, neurons_list_path)
    neuronal_names_ordered_by_type = []
    for neuronal_type in neuronal_types_by_names:
        neuronal_names_ordered_by_type += sorted(list(neuronal_type))

    with open(neurons_list_path, 'rb') as f:
        neurons_list = pickle.load(f)

    single_color = SINGLE_EPOCH_INFERRED_COLOR
    multiple_color = adjust_lightness('lightcoral', 1.0)
    padding = 0.05
    single_axis_size = (1 - len(worms_indices) * padding) / len(worms_indices)
    fig_ratio = (3 * padding + 3 * single_axis_size)
    fig_width = 4 * len(worms_indices)
    fig1 = plt.figure(1, figsize=(fig_width * CM_TO_INCH, fig_ratio * fig_width * CM_TO_INCH))
    axs = fig1.subplots(3, len(worms_indices))
    vmax = 1
    fontsize = FONT_SIZE
    tick_labelpad = 1
    axes_ticks = range(0, 161, 80)
    x_ticks_rot = 90
    cur_col = 0

    for i in worms_indices:
        cur_data_path = os.path.join(data_dir_path, f'Dataset{i}.pkl')
        cur_age = FULL_DEVELOPMENTAL_AGES[min(i - 1, 6)]
        cur_multiple_dyads_train_path = os.path.join(dyads_dist_multiple_epochs_dir_path, "TrainSet",
                                                     f'split{split}', f'{smi_mulitple:.5f}',
                                                     f'{cur_age}.pkl')
        cur_multiple_dyads_test_path = os.path.join(dyads_dist_multiple_epochs_dir_path, "TestSet",
                                                    f'split{split}', f'{smi_mulitple:.5f}',
                                                    f'{cur_age}.pkl')
        cur_single_dyads_train_path = os.path.join(dyads_dist_single_epoch_dir_path, "TrainSet", f'split{split}',
                                                   f'{smi_single:.5f}',
                                                   f'{cur_age}.pkl')
        cur_single_dyads_test_path = os.path.join(dyads_dist_single_epoch_dir_path, "TestSet", f'split{split}',
                                                  f'{smi_single:.5f}',
                                                  f'{cur_age}.pkl')

        with open(cur_data_path, 'rb') as f:
            cur_data = pickle.load(f)
        cur_alphabetic_neuronal_names = sorted(cur_data.nodes)
        cur_data_mat = nx.to_numpy_array(cur_data, nodelist=cur_alphabetic_neuronal_names)
        cur_neurons_names_by_type = neuronal_names_ordered_by_type.copy()
        for n in neurons_list:
            if n not in cur_alphabetic_neuronal_names:
                cur_neurons_names_by_type.remove(n)

        cur_neurons_idx_by_type = np.zeros((len(cur_alphabetic_neuronal_names), 1)).astype(int)
        for j in range(len(cur_alphabetic_neuronal_names)):
            cur_neurons_idx_by_type[j] = cur_alphabetic_neuronal_names.index(cur_neurons_names_by_type[j])

        cur_data_mat = cur_data_mat[cur_neurons_idx_by_type, cur_neurons_idx_by_type.T]
        cur_ax = axs[0, cur_col]
        cur_ax.imshow(cur_data_mat, cmap=colors.LinearSegmentedColormap.from_list('data', [(1, 1, 1), (0, 0, 0)]),
                      vmin=0, vmax=vmax)
        if i > 1:
            cur_ax.set_yticks(axes_ticks)
            cur_ax.set_yticklabels([])
        else:
            cur_ax.set_yticks(axes_ticks)
            cur_ax.yaxis.set_tick_params(pad=tick_labelpad)
            cur_ax.set_yticklabels([f'{tick}' for tick in axes_ticks], fontsize=fontsize)
        cur_ax.set_xticks(axes_ticks)
        cur_ax.set_xticklabels([])

        cur_single_mat = sample_from_dyads_distribution_train_test(cur_single_dyads_train_path,
                                                                   cur_single_dyads_test_path,
                                                                   cur_alphabetic_neuronal_names)
        cur_single_mat = cur_single_mat[cur_neurons_idx_by_type, cur_neurons_idx_by_type.T]
        cur_ax = axs[1, cur_col]
        cur_ax.imshow(cur_single_mat,
                      cmap=colors.LinearSegmentedColormap.from_list('single_epoch', [(1, 1, 1), single_color]),
                      vmin=0, vmax=vmax)
        if i > 1:
            cur_ax.set_yticks(axes_ticks)
            cur_ax.set_yticklabels([])
        else:
            cur_ax.set_yticks(axes_ticks)
            cur_ax.yaxis.set_tick_params(pad=tick_labelpad)
            cur_ax.set_yticklabels([f'{tick}' for tick in axes_ticks], fontsize=fontsize)
        cur_ax.set_xticks(axes_ticks)
        cur_ax.set_xticklabels([])

        cur_multiple_mat = sample_from_dyads_distribution_train_test(cur_multiple_dyads_train_path,
                                                                     cur_multiple_dyads_test_path,
                                                                     cur_alphabetic_neuronal_names)
        cur_multiple_mat = cur_multiple_mat[cur_neurons_idx_by_type, cur_neurons_idx_by_type.T]
        cur_ax = axs[2, cur_col]
        cur_ax.imshow(cur_multiple_mat,
                      cmap=colors.LinearSegmentedColormap.from_list('multiple_color',
                                                                    [(1, 1, 1),
                                                                     multiple_color]),
                      vmin=0, vmax=vmax)
        if i > 1:
            cur_ax.set_yticks(axes_ticks)
            cur_ax.set_yticklabels([])
        else:
            cur_ax.set_yticks(axes_ticks)
            cur_ax.yaxis.set_tick_params(pad=tick_labelpad)
            cur_ax.set_yticklabels([f'{tick}' for tick in axes_ticks], fontsize=fontsize)
        cur_ax.set_xticks(axes_ticks)
        cur_ax.xaxis.set_tick_params(pad=tick_labelpad)
        cur_ax.set_xticklabels([f'{tick}' for tick in axes_ticks], fontsize=fontsize, rotation=x_ticks_rot)

        cur_col += 1

    plt.subplots_adjust(left=padding, bottom=padding / fig_ratio, right=0.99, top=0.99, wspace=padding,
                        hspace=padding / fig_ratio)


def fig_6_a(out_path="Figures\\Fig6", split=16):
    _fig_6_a([1, 3, 5, 8], split=split)
    plt.savefig(os.path.join(out_path, '6_a.pdf'), format='pdf')
    plt.show()

    _fig_6_a(list(range(1, 7)) + [8], split=split)
    plt.savefig(os.path.join(out_path, '6_a_all_worms.pdf'), format='pdf')
    plt.show()


def fig_6_b(out_path="Figures\\Fig6\\b"):
    with open("SavedOutputs\ReciprocalModel\\DyadsSplit\\max_likelihood_params_per_split_single_epoch.pkl",
              'rb') as f:
        max_like_params_single = pickle.load(f)
    with open("SavedOutputs\ReciprocalModel\\DyadsSplit\\max_likelihood_params_per_split_3_epochs.pkl",
              'rb') as f:
        max_like_params_multiple = pickle.load(f)
    num_splits = 20
    for split in range(1, num_splits + 1):
        smi_single = max_like_params_single[f'split{split}']['S-']
        smi_multiple = max_like_params_multiple[f'split{split}']['S-']
        single_model_dyads_test_path = f"SavedOutputs\ReciprocalModel\\DyadsSplit\\dyads_distributions\SingleDevStage\TestSet\\split{split}\\{smi_single:.5f}\\3500.pkl"
        multiple_model_dyads_test_path = f"SavedOutputs\ReciprocalModel\\DyadsSplit\\dyads_distributions\ThreeDevStages\TestSet\\split{split}\\{smi_multiple:.5f}\\3500.pkl"
        test_data_path = f"CElegansData\InferredTypes\\synapses_lists\8_types\\split{split}\\test\Dataset8.pkl"
        with open(test_data_path, 'rb') as f:
            test_data = sorted(pickle.load(f))
        test_data_exists = []
        for syn in test_data:
            test_data_exists.append(syn[-1]['exists'])

        with open(single_model_dyads_test_path, 'rb') as f:
            single_model_dyads_dists = pickle.load(f)
        with open(multiple_model_dyads_test_path, 'rb') as f:
            multiple_model_dyads_dists = pickle.load(f)

        single_model_probs = construct_probs_array_from_dyads_dist_string_keys(single_model_dyads_dists)
        multiple_model_probs = construct_probs_array_from_dyads_dist_string_keys(multiple_model_dyads_dists)

        single_auc = roc_auc_score(test_data_exists, single_model_probs)
        single_false_positive_rate, single_true_positive_rate, _ = roc_curve(test_data_exists, single_model_probs)

        multiple_auc = roc_auc_score(test_data_exists, multiple_model_probs)
        multiple_false_positive_rate, multiple_true_positive_rate, _ = roc_curve(test_data_exists, multiple_model_probs)

        fig = plt.figure(figsize=SQUARE_FIG_SIZE)
        fontsize = FONT_SIZE
        markersize = MARKER_SIZE
        line_width = LINE_WIDTH
        axes_labelpad = 1
        main_axes = [0.2, 0.175, 0.78, 0.78]
        ax1 = fig.add_axes(main_axes)
        ax1.set_aspect('equal', 'box')
        axes_ticks = np.arange(0, 1.2, 0.5)
        ax1.set_xticks(axes_ticks)
        ax1.set_xticklabels([f'{tick:.1f}' for tick in axes_ticks], fontsize=fontsize)
        ax1.set_yticks(axes_ticks)
        ax1.set_yticklabels([f'{tick:.1f}' for tick in axes_ticks], fontsize=fontsize)
        ax1.plot([0, 1], [0, 1], 'gray', lw=LINE_WIDTH)
        ax1.set_xlabel('False Positive Rate', fontsize=fontsize, labelpad=axes_labelpad)
        ax1.set_ylabel('True Positive Rate', fontsize=fontsize, labelpad=axes_labelpad)

        single_color = SINGLE_EPOCH_INFERRED_COLOR
        multiple_color = adjust_lightness('lightcoral', 1.0)

        ax1.plot(single_false_positive_rate,
                 single_true_positive_rate, marker='.',
                 label=f"single epoch, AUC={single_auc:.2f}", c=single_color, markersize=3 * markersize, lw=line_width)
        ax1.plot(multiple_false_positive_rate, multiple_true_positive_rate, marker='.',
                 label=f"multiple epochs, AUC={multiple_auc:.2f}", c=multiple_color, markersize=markersize,
                 lw=line_width)

        plt.savefig(os.path.join(out_path, f'single_multiple_rocs_split{split}.pdf'), format='pdf')
        plt.show()


def fig_6_d(out_path="Figures\\Fig6\\d"):
    num_stds = 1
    train_or_test = 'Test'
    with open("SavedOutputs\ReciprocalModel\\DyadsSplit\\max_likelihood_params_per_split_single_epoch.pkl",
              'rb') as f:
        max_like_params_single = pickle.load(f)
    with open("SavedOutputs\ReciprocalModel\\DyadsSplit\\max_likelihood_params_per_split_3_epochs.pkl",
              'rb') as f:
        max_like_params_multiple = pickle.load(f)
    num_splits = 20
    single_pruning_density = np.zeros((num_splits, ADULT_WORM_AGE // 10))
    multiple_pruning_density = np.zeros((num_splits, ADULT_WORM_AGE // 10))
    data_ages = np.array(
        [FULL_DEVELOPMENTAL_AGES[stage] for stage in sorted(FULL_DEVELOPMENTAL_AGES.keys())])
    data_density = np.zeros((num_splits, data_ages.size))
    for split in range(1, num_splits + 1):
        smi_single = max_like_params_single[f'split{split}']["S-"]
        smi_multiple = max_like_params_multiple[f'split{split}']["S-"]
        single_pruning_path = f"SavedOutputs\ReciprocalModel\\DyadsSplit\dyads_distributions\SingleDevStage\\{train_or_test}Set\\split{split}\\{smi_single:.5f}"
        multiple_pruning_path = f"SavedOutputs\ReciprocalModel\\DyadsSplit\\dyads_distributions\ThreeDevStages\\{train_or_test}Set\\split{split}\\{smi_multiple:.5f}"
        data_path = f"CElegansData\InferredTypes\\synapses_lists\8_types\\split{split}\\{train_or_test.lower()}"

        single_pruning_density_std = np.zeros(ADULT_WORM_AGE // 10)
        multiple_pruning_density_std = np.zeros(ADULT_WORM_AGE // 10)

        for dyads_dist_file in os.listdir(single_pruning_path):
            cur_age = int(dyads_dist_file[:dyads_dist_file.find('.pkl')])
            cur_idx = cur_age // 10 - 1
            with open(os.path.join(single_pruning_path, dyads_dist_file), 'rb') as f:
                single_pruning_dyads_dist = pickle.load(f)
            if len(single_pruning_dyads_dist.keys()) == 0:
                continue
            single_pruning_density[split - 1, cur_idx], single_pruning_density_std[
                cur_idx] = calc_reciprocal_dependence_model_density_from_dyads_dist_string_keys(
                single_pruning_dyads_dist)

            with open(os.path.join(multiple_pruning_path, dyads_dist_file), 'rb') as f:
                multiple_pruning_dyads_dist = pickle.load(f)
            multiple_pruning_density[split - 1, cur_idx], multiple_pruning_density_std[
                cur_idx] = calc_reciprocal_dependence_model_density_from_dyads_dist_string_keys(
                multiple_pruning_dyads_dist)

            if cur_age in data_ages:
                dataset_idx = list(data_ages).index(cur_age) + 1
                if dataset_idx == 7:
                    dataset_idx = 8
                data_exists_array = construct_exists_array_from_syn_list(
                    os.path.join(data_path, f"Dataset{dataset_idx}.pkl"))
                data_density[split - 1, min(dataset_idx - 1, 6)] = data_exists_array.mean()

        min_plotted_age = 300
        ages = np.arange(min_plotted_age, ADULT_WORM_AGE + 10, 10)
        single_pruning_density_for_plot = single_pruning_density[split - 1, min_plotted_age // 10 - 1:]
        single_pruning_density_std_for_plot = single_pruning_density_std[min_plotted_age // 10 - 1:]
        multiple_pruning_density_for_plot = multiple_pruning_density[split - 1, min_plotted_age // 10 - 1:]
        multiple_pruning_density_std_for_plot = multiple_pruning_density_std[min_plotted_age // 10 - 1:]

        fontsize = FONT_SIZE
        markersize = MARKER_SIZE
        line_width = LINE_WIDTH
        main_axes = [0.17, 0.175, 0.8, 0.8]
        axes_labelpad = 1
        xticks = range(min_plotted_age, ADULT_WORM_AGE + 100, 800)
        yticks = np.arange(0, 0.1, 0.02)
        fig1 = plt.figure(1, figsize=RECT_LARGE_FIG_SIZE)
        ax1 = fig1.add_axes(main_axes)
        ax1.set_xticks(xticks)
        ax1.set_xticklabels([str(i) for i in xticks], fontsize=fontsize)
        ax1.set_yticks(yticks)
        ax1.set_yticklabels([f"{i:.2f}" for i in yticks], fontsize=fontsize)
        ax1.set_xlim(min_plotted_age - 50, ADULT_WORM_AGE + 100)
        ax1.set_ylim(0, 0.09)
        ax1.set_xlabel('worm age [min.]', fontsize=fontsize, labelpad=axes_labelpad)
        ax1.set_ylabel('network density', fontsize=fontsize, labelpad=axes_labelpad)

        single_pruning_color = SINGLE_EPOCH_INFERRED_COLOR
        multiple_pruning_color = adjust_lightness('lightcoral', 1.0)
        alpha_value = 0.3

        ax1.plot(ages, single_pruning_density_for_plot, color=single_pruning_color, markersize=markersize,
                 lw=line_width,
                 label='single epoch pruning')
        ax1.fill_between(ages, single_pruning_density_for_plot + num_stds * single_pruning_density_std_for_plot,
                         np.clip(single_pruning_density_for_plot - num_stds * single_pruning_density_std_for_plot, 0,
                                 None),
                         color=single_pruning_color, alpha=alpha_value)
        ax1.plot(ages, multiple_pruning_density_for_plot, color=multiple_pruning_color, markersize=markersize,
                 lw=line_width,
                 label='multiple epochs pruning')
        ax1.fill_between(ages, multiple_pruning_density_for_plot + num_stds * multiple_pruning_density_std_for_plot,
                         np.clip(multiple_pruning_density_for_plot - num_stds * multiple_pruning_density_std_for_plot,
                                 0, None),
                         color=multiple_pruning_color, alpha=alpha_value)
        ax1.plot(data_ages, data_density[split - 1], '.', c='k', markersize=markersize * 1.5, label='data')
        plt.savefig(os.path.join(out_path, f'6_d_split{split}.pdf'), format='pdf')
        plt.show(block=False)
        plt.pause(3)
        plt.close()

    fig1 = plt.figure(1, figsize=RECT_LARGE_FIG_SIZE)
    ax1 = fig1.add_axes(main_axes)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels([str(i) for i in xticks], fontsize=fontsize)
    ax1.set_yticks(yticks)
    ax1.set_yticklabels([f"{i:.2f}" for i in yticks], fontsize=fontsize)
    ax1.set_xlim(min_plotted_age - 50, ADULT_WORM_AGE + 100)
    ax1.set_ylim(0, 0.09)
    ax1.set_xlabel('worm age [min.]', fontsize=fontsize, labelpad=axes_labelpad)
    ax1.set_ylabel('network density', fontsize=fontsize, labelpad=axes_labelpad)

    single_pruning_color = SINGLE_EPOCH_INFERRED_COLOR
    multiple_pruning_color = adjust_lightness('lightcoral', 1.0)
    alpha_value = 0.3

    single_pruning_density_mean_across_splits = single_pruning_density.mean(axis=0)[min_plotted_age // 10 - 1:]
    single_pruning_density_std_across_splits = single_pruning_density.std(axis=0)[min_plotted_age // 10 - 1:]
    multiple_pruning_density_mean_across_splits = multiple_pruning_density.mean(axis=0)[min_plotted_age // 10 - 1:]
    multiple_pruning_density_std_across_splits = multiple_pruning_density.std(axis=0)[min_plotted_age // 10 - 1:]
    test_data_density_mean_acrosss_splits = data_density.mean(axis=0)
    test_data_density_std_acrosss_splits = data_density.std(axis=0)

    ax1.plot(ages, single_pruning_density_mean_across_splits, color=single_pruning_color, markersize=markersize,
             lw=line_width,
             label='single epoch pruning')
    ax1.fill_between(ages,
                     single_pruning_density_mean_across_splits + num_stds * single_pruning_density_std_across_splits,
                     np.clip(
                         single_pruning_density_mean_across_splits - num_stds * single_pruning_density_std_across_splits,
                         0, None),
                     color=single_pruning_color, alpha=alpha_value)
    ax1.plot(ages, multiple_pruning_density_mean_across_splits, color=multiple_pruning_color, markersize=markersize,
             lw=line_width,
             label='multiple epochs pruning')
    ax1.fill_between(ages,
                     multiple_pruning_density_mean_across_splits + num_stds * multiple_pruning_density_std_across_splits,
                     np.clip(
                         multiple_pruning_density_mean_across_splits - num_stds * multiple_pruning_density_std_across_splits,
                         0,
                         None),
                     color=multiple_pruning_color, alpha=alpha_value)
    ax1.errorbar(data_ages, test_data_density_mean_acrosss_splits, yerr=test_data_density_std_acrosss_splits,
                 marker='o', ls='none', c='k', markersize=markersize * 1.5, label='data')

    plt.savefig(os.path.join(out_path, f'6_d_average_across_splits.pdf'), format='pdf')
    plt.show(block=False)
    plt.pause(3)
    plt.close()


def fig_6_e(out_path="Figures\\Fig6\\e", saved_calcs_path="Figures\SavedCalcs", is_saved=False, do_save=True):
    num_splits = 20
    train_or_test = "Test"
    data_ages = np.array(
        [FULL_DEVELOPMENTAL_AGES[stage] for stage in sorted(FULL_DEVELOPMENTAL_AGES.keys())])
    with open("SavedOutputs\ReciprocalModel\\DyadsSplit\\max_likelihood_params_per_split_single_epoch.pkl",
              'rb') as f:
        max_like_params_single = pickle.load(f)
    with open("SavedOutputs\ReciprocalModel\\DyadsSplit\\max_likelihood_params_per_split_3_epochs.pkl",
              'rb') as f:
        max_like_params_multiple = pickle.load(f)
    if not is_saved:
        single_pruning_likelihoods = np.zeros((num_splits, data_ages.size))
        multiple_pruning_likelihoods = np.zeros((num_splits, data_ages.size))
        for split in range(1, num_splits + 1):
            single_smi = max_like_params_single[f'split{split}']['S-']
            multiple_smi = max_like_params_multiple[f'split{split}']['S-']
            single_pruning_path = f"SavedOutputs\ReciprocalModel\\DyadsSplit\\dyads_distributions\SingleDevStage\{train_or_test}Set\\split{split}\\{single_smi:.5f}"
            multiple_pruning_path = f"SavedOutputs\ReciprocalModel\\DyadsSplit\\dyads_distributions\\ThreeDevStages\{train_or_test}Set\\split{split}\\{multiple_smi:.5f}"
            data_path = f"CElegansData\InferredTypes\\synapses_lists\8_types\\split{split}\\{train_or_test.lower()}"

            cur_idx = 0
            for age in data_ages:
                data_set_idx = cur_idx + 1
                if data_set_idx == 7:
                    data_set_idx = 8
                with open(os.path.join(single_pruning_path, f'{int(age)}.pkl'), 'rb') as f:
                    single_dyads_distributions = pickle.load(f)
                with open(os.path.join(multiple_pruning_path, f'{int(age)}.pkl'), 'rb') as f:
                    multiple_dyads_distributions = pickle.load(f)
                single_pruning_likelihoods[
                    split - 1, cur_idx] = calc_reciprocal_dependence_model_log_likelihood_from_dyads_dist_string_keys(
                    single_dyads_distributions, os.path.join(data_path, f'Dataset{data_set_idx}.pkl'))

                multiple_pruning_likelihoods[
                    split - 1, cur_idx] = calc_reciprocal_dependence_model_log_likelihood_from_dyads_dist_string_keys(
                    multiple_dyads_distributions, os.path.join(data_path, f'Dataset{data_set_idx}.pkl'))
                cur_idx += 1

    else:
        with open(os.path.join(saved_calcs_path, "likelihoods_single_stage_across_ages_and_splits.pkl"), 'rb') as f:
            single_pruning_likelihoods = pickle.load(f)
        with open(os.path.join(saved_calcs_path, "likelihoods_three_stages_across_ages_and_splits.pkl"), 'rb') as f:
            multiple_pruning_likelihoods = pickle.load(f)

    for split in range(1, num_splits + 1):
        fontsize = FONT_SIZE
        markersize = MARKER_SIZE
        main_axes = [0.18, 0.175, 0.8, 0.8]
        axes_labelpad = 1
        xticks = range(700, ADULT_WORM_AGE + 100, 700)
        yticks = range(-20, 41, 20)
        fig1 = plt.figure(1, figsize=SQUARE_FIG_SIZE)
        ax1 = fig1.add_axes(main_axes)
        ax1.set_xticks(xticks)
        ax1.set_xticklabels([str(i) for i in xticks], fontsize=fontsize)
        ax1.set_yticks(yticks)
        ax1.set_yticklabels([str(i) for i in yticks], fontsize=fontsize)
        ax1.set_xlim(FULL_DEVELOPMENTAL_AGES[0] - 200, ADULT_WORM_AGE + 200)
        ax1.set_ylim(-23.5, 46)
        ax1.set_xlabel('worm age [min.]', fontsize=fontsize, labelpad=axes_labelpad)
        ax1.set_ylabel(r'$\log _{10} ( \mathcal{L} _{\mathrm{multi}} / \mathcal{L} _{\mathrm{single}} )$',
                       fontsize=fontsize,
                       labelpad=axes_labelpad - 5)

        ax1.plot([FULL_DEVELOPMENTAL_AGES[0] - 100, ADULT_WORM_AGE + 100], [0, 0],
                 c=adjust_lightness('grey', 1.5), markersize=markersize)
        ax1.plot(data_ages, np.log10(np.exp(1)) * (
                multiple_pruning_likelihoods[split - 1] - single_pruning_likelihoods[split - 1]), '.', c='k',
                 markersize=markersize * 3)

        plt.savefig(os.path.join(out_path, f'likelihood_split{split}_{train_or_test}.pdf'), format='pdf')
        plt.show(block=False)
        plt.pause(3)
        plt.close()

    if do_save:
        with open(os.path.join(saved_calcs_path, "likelihoods_single_stage_across_ages_and_splits.pkl"), 'wb') as f:
            pickle.dump(single_pruning_likelihoods, f)
        with open(os.path.join(saved_calcs_path, "likelihoods_three_stages_across_ages_and_splits.pkl"), 'wb') as f:
            pickle.dump(multiple_pruning_likelihoods, f)

    fig1 = plt.figure(1, figsize=SQUARE_FIG_SIZE)
    ax1 = fig1.add_axes(main_axes)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels([str(i) for i in xticks], fontsize=fontsize)
    ax1.set_yticks(yticks)
    ax1.set_yticklabels([str(i) for i in yticks], fontsize=fontsize)
    ax1.set_xlim(FULL_DEVELOPMENTAL_AGES[0] - 200, ADULT_WORM_AGE + 200)
    ax1.set_ylim(-23.5, 46)
    ax1.set_xlabel('worm age [min.]', fontsize=fontsize, labelpad=axes_labelpad)
    ax1.set_ylabel(r'$\log _{10} ( \mathcal{L} _{\mathrm{multi}} / \mathcal{L} _{\mathrm{single}} )$',
                   fontsize=fontsize,
                   labelpad=axes_labelpad - 5)

    ax1.plot([FULL_DEVELOPMENTAL_AGES[0] - 100, ADULT_WORM_AGE + 100], [0, 0],
             c=adjust_lightness('grey', 1.5), markersize=markersize)
    ax1.errorbar(data_ages,
                 np.log10(np.exp(1)) * (multiple_pruning_likelihoods - single_pruning_likelihoods).mean(axis=0),
                 yerr=(np.log10(np.exp(1)) * (multiple_pruning_likelihoods - single_pruning_likelihoods)).std(axis=0),
                 marker='o', ls='none', c='k', markersize=markersize * 3)

    plt.savefig(os.path.join(out_path, f'average_likelihoods_across_splits_{train_or_test}.pdf'), format='pdf')
    plt.show(block=False)
    plt.pause(3)
    plt.close()


def fig_6_f(out_path="Figures\\Fig6\\f", saved_calcs_path="Figures\SavedCalcs", is_saved=False, do_save=True):
    num_splits = 20
    data_ages = np.array(
        [FULL_DEVELOPMENTAL_AGES[stage] for stage in sorted(FULL_DEVELOPMENTAL_AGES.keys())])
    train_or_test = "Test"
    if not is_saved:
        with open("SavedOutputs\ReciprocalModel\\DyadsSplit\\max_likelihood_params_per_split_3_epochs.pkl",
                  'rb') as f:
            max_like_params_multiple = pickle.load(f)

        multiple_pruning_likelihoods = np.zeros((num_splits, data_ages.size))
        multiple_no_pruning_likelihoods = np.zeros((num_splits, data_ages.size))
        for split in range(1, num_splits + 1):
            smi = max_like_params_multiple[f'split{split}']['S-']
            multiple_no_pruning_path = f"SavedOutputs\ReciprocalModel\\DyadsSplit\\dyads_distributions\ThreeDevStages\\{train_or_test}Set\\split{split}\\{0:.5f}"
            multiple_pruning_path = f"SavedOutputs\ReciprocalModel\\DyadsSplit\\dyads_distributions\ThreeDevStages\\{train_or_test}Set\\split{split}\\{smi:.5f}"
            data_path = f"CElegansData\InferredTypes\\synapses_lists\8_types\\split{split}\\{train_or_test.lower()}"

            cur_idx = 0
            for age in data_ages:
                data_set_idx = cur_idx + 1
                if data_set_idx == 7:
                    data_set_idx = 8
                with open(os.path.join(multiple_no_pruning_path, f'{int(age)}.pkl'), 'rb') as f:
                    multiple_no_pruning_dyads_distributions = pickle.load(f)
                with open(os.path.join(multiple_pruning_path, f'{int(age)}.pkl'), 'rb') as f:
                    multiple_dyads_distributions = pickle.load(f)
                multiple_no_pruning_likelihoods[
                    split - 1, cur_idx] = calc_reciprocal_dependence_model_log_likelihood_from_dyads_dist_string_keys(
                    multiple_no_pruning_dyads_distributions, os.path.join(data_path, f'Dataset{data_set_idx}.pkl'))

                multiple_pruning_likelihoods[
                    split - 1, cur_idx] = calc_reciprocal_dependence_model_log_likelihood_from_dyads_dist_string_keys(
                    multiple_dyads_distributions, os.path.join(data_path, f'Dataset{data_set_idx}.pkl'))
                cur_idx += 1

    else:
        with open(os.path.join(saved_calcs_path, "likelihoods_three_stages_across_ages_and_splits.pkl"), 'rb') as f:
            multiple_pruning_likelihoods = pickle.load(f)
        with open(os.path.join(saved_calcs_path, "likelihoods_three_stages_no_pruning_across_ages_and_splits.pkl"),
                  'rb') as f:
            multiple_no_pruning_likelihoods = pickle.load(f)
    for split in range(1, num_splits + 1):
        fontsize = FONT_SIZE
        markersize = MARKER_SIZE
        main_axes = [0.18, 0.175, 0.8, 0.8]
        axes_labelpad = 1
        xticks = range(700, ADULT_WORM_AGE + 100, 700)
        yticks = range(-15, 31, 15)
        fig1 = plt.figure(1, figsize=SQUARE_FIG_SIZE)
        ax1 = fig1.add_axes(main_axes)
        ax1.set_xticks(xticks)
        ax1.set_xticklabels([str(i) for i in xticks], fontsize=fontsize)
        ax1.set_yticks(yticks)
        ax1.set_yticklabels([str(i) for i in yticks], fontsize=fontsize)
        ax1.set_xlim(FULL_DEVELOPMENTAL_AGES[0] - 200, ADULT_WORM_AGE + 200)
        ax1.set_ylim(-16, 31)
        ax1.set_xlabel('worm age [min.]', fontsize=fontsize, labelpad=axes_labelpad)
        ax1.set_ylabel(r'$\log _{10} ( \mathcal{L} _{\mathrm{pruning}} / \mathcal{L} _{\mathrm{no \: pruning}} )$',
                       fontsize=fontsize,
                       labelpad=axes_labelpad - 5)

        ax1.plot([FULL_DEVELOPMENTAL_AGES[0] - 100, ADULT_WORM_AGE + 100], [0, 0],
                 c=adjust_lightness('grey', 1.5), markersize=markersize)
        ax1.plot(data_ages, np.log10(np.exp(1)) * (
                multiple_pruning_likelihoods[split - 1] - multiple_no_pruning_likelihoods[split - 1]), '.',
                 c='k', markersize=markersize * 3)

        plt.savefig(os.path.join(out_path, f'likelihood_split{split}_{train_or_test}.pdf'), format='pdf')
        plt.show(block=False)
        plt.pause(3)
        plt.close()

    if do_save:
        with open(os.path.join(saved_calcs_path, "likelihoods_three_stages_no_pruning_across_ages_and_splits.pkl"),
                  'wb') as f:
            pickle.dump(multiple_no_pruning_likelihoods, f)
        with open(os.path.join(saved_calcs_path, "likelihoods_three_stages_across_ages_and_splits.pkl"), 'wb') as f:
            pickle.dump(multiple_pruning_likelihoods, f)

    fig1 = plt.figure(1, figsize=SQUARE_FIG_SIZE)
    ax1 = fig1.add_axes(main_axes)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels([str(i) for i in xticks], fontsize=fontsize)
    ax1.set_yticks(yticks)
    ax1.set_yticklabels([str(i) for i in yticks], fontsize=fontsize)
    ax1.set_xlim(FULL_DEVELOPMENTAL_AGES[0] - 200, ADULT_WORM_AGE + 200)
    ax1.set_ylim(-15, 31)
    ax1.set_xlabel('worm age [min.]', fontsize=fontsize, labelpad=axes_labelpad)
    ax1.set_ylabel(r'$\log _{10} ( \mathcal{L} _{\mathrm{pruning}} / \mathcal{L} _{\mathrm{no \: pruning}} )$',
                   fontsize=fontsize,
                   labelpad=axes_labelpad - 5)

    ax1.plot([FULL_DEVELOPMENTAL_AGES[0] - 100, ADULT_WORM_AGE + 100], [0, 0],
             c=adjust_lightness('grey', 1.5), markersize=markersize)
    ax1.errorbar(data_ages,
                 np.log10(np.exp(1)) * (multiple_pruning_likelihoods - multiple_no_pruning_likelihoods).mean(axis=0),
                 yerr=(np.log10(np.exp(1)) * (multiple_pruning_likelihoods - multiple_no_pruning_likelihoods)).std(
                     axis=0), marker='o', ls='none', c='k', markersize=markersize * 3)

    plt.savefig(os.path.join(out_path, f'average_likelihoods_across_splits_{train_or_test}.pdf'), format='pdf')
    plt.show(block=False)
    plt.pause(3)
    plt.close()


def worms_overlap(out_path=os.path.join("Figures", "Fig7")):
    def _plot_adj_mat(m, out_file_name=None):
        data_cmap = colors.LinearSegmentedColormap.from_list('data', [(1, 1, 1), (0, 0, 0)])
        pylab.rcParams['xtick.major.pad'] = '0.5'
        pylab.rcParams['ytick.major.pad'] = '0.5'
        axis_labelpad = 1
        fig = plt.figure(figsize=SQUARE_FIG_SIZE)
        ax = fig.add_axes([0.18, 0.18, 0.75, 0.75])

        neurons_idx_by_type = _get_neurons_idx_by_inferred_type(num_types=8)

        im = ax.imshow(m[neurons_idx_by_type, neurons_idx_by_type.T], cmap=data_cmap)
        axis_ticks = range(0, 181, 60)
        ax.set_xlabel("post-synaptic neuronal idx", fontsize=FONT_SIZE, labelpad=axis_labelpad)
        ax.set_ylabel("pre-synaptic neuronal idx", fontsize=FONT_SIZE, labelpad=axis_labelpad)
        ax.set_xticks(axis_ticks)
        ax.set_xticklabels([str(i) for i in axis_ticks], fontsize=FONT_SIZE)
        ax.set_yticks(axis_ticks)
        ax.set_yticklabels([str(i) for i in axis_ticks], fontsize=FONT_SIZE)

        if out_file_name is not None:
            plt.savefig(os.path.join(out_path, out_file_name), format='pdf')
        plt.show()

    worm_7_path = os.path.join('.', "CElegansData", "InferredTypes", "connectomes", "8_types", "Dataset7.pkl")
    with open(worm_7_path, 'rb') as f:
        worm_7 = pickle.load(f)
    worm_8_path = os.path.join('.', "CElegansData", "InferredTypes", "connectomes", "8_types", "Dataset8.pkl")
    with open(worm_8_path, 'rb') as f:
        worm_8 = pickle.load(f)
    worm_atlas_path = os.path.join('.', 'CElegansData',
                                   'worm_atlas_sub_connectome_chemical_no_autosynapses_subtypes.pkl')
    with open(worm_atlas_path, 'rb') as f:
        worm_atlas = pickle.load(f)

    nerve_ring_neurons = sorted(list(worm_7.nodes()))

    worm_7_adj_mat = nx.to_numpy_array(worm_7, nodelist=nerve_ring_neurons)
    worm_8_adj_mat = nx.to_numpy_array(worm_8, nodelist=nerve_ring_neurons)
    worm_atlas_adj_mat = nx.to_numpy_array(worm_atlas, nodelist=nerve_ring_neurons)

    matrices_to_plot = [worm_8_adj_mat, worm_7_adj_mat * worm_8_adj_mat,
                        worm_7_adj_mat * worm_8_adj_mat * worm_atlas_adj_mat]
    file_names = ["worm_8.pdf", "mei_zhen_overlap.pdf", "backbone.pdf"]
    for m, f in zip(matrices_to_plot, file_names):
        _plot_adj_mat(m, out_file_name=f)


def supplement_model_variance(out_path=os.path.join("Figures", "FigS_multi_epochs_across_splits_7", "a"),
                              saved_calcs_path=os.path.join("Figures", "SavedCalcs"),
                              is_saved=False, do_save=True):
    num_splits = 20
    data_ages = np.array(
        [FULL_DEVELOPMENTAL_AGES[stage] for stage in sorted(FULL_DEVELOPMENTAL_AGES.keys())])
    train_or_test = 'Test'
    with open(os.path.join("SavedOutputs", "ReciprocalModel", "DyadsSplit",
                           "max_likelihood_params_per_split_3_epochs.pkl"), 'rb') as f:
        max_like_params_multiple = pickle.load(f)
    if not is_saved:
        model_variance = np.zeros((num_splits, ADULT_WORM_AGE // 10))
        model_variance_std = np.zeros((num_splits, ADULT_WORM_AGE // 10))
        for split in tqdm(range(1, num_splits + 1)):
            smi_multiple = max_like_params_multiple[f'split{split}']['S-']
            model_test_dyads_path = os.path.join("SavedOutputs", "ReciprocalModel", "DyadsSplit", "dyads_distributions",
                                                 "ThreeDevStages", f"{train_or_test}Set", f"split{split}",
                                                 f"{smi_multiple:.5f}")
            for dyads_file in os.listdir(model_test_dyads_path):
                with open(os.path.join(model_test_dyads_path, dyads_file), 'rb') as f:
                    dyads_dist = pickle.load(f)
                cur_age = int(dyads_file[:dyads_file.find('.pkl')])
                cur_idx = cur_age // 10 - 1

                model_variance[split - 1, cur_idx], model_variance_std[
                    split - 1, cur_idx] = calc_reciprocal_dependence_model_variance_from_dyads_dist_str_keys(dyads_dist)

        model_data_cross_variances = np.zeros((num_splits, len(FULL_DEVELOPMENTAL_AGES.keys())))
        model_data_cross_variances_std = np.zeros((num_splits, len(FULL_DEVELOPMENTAL_AGES.keys())))
        for split in tqdm(range(1, num_splits + 1)):
            test_data_path = os.path.join(f"CElegansData", "InferredTypes", "synapses_lists", "8_types",
                                          f"split{split}",
                                          f"{train_or_test.lower()}")
            smi_multiple = max_like_params_multiple[f'split{split}']['S-']
            model_test_dyads_path = os.path.join("SavedOutputs", "ReciprocalModel", "DyadsSplit", "dyads_distributions",
                                                 "ThreeDevStages", f"{train_or_test}Set", f"split{split}",
                                                 f"{smi_multiple:.5f}")
            for data_age in data_ages:
                dataset_idx = list(data_ages).index(data_age) + 1
                if dataset_idx == 7:
                    dataset_idx = 8
                with open(os.path.join(model_test_dyads_path, f'{data_age}.pkl'), 'rb') as f:
                    dyads_dist = pickle.load(f)
                model_data_cross_variances[split - 1, min(dataset_idx - 1, 6)], model_data_cross_variances_std[
                    split - 1, min(dataset_idx - 1,
                                   6)] = calc_reciprocal_dependence_model_data_cross_variance_from_dyads_dist_str_keys(
                    dyads_dist, os.path.join(test_data_path, f"Dataset{dataset_idx}.pkl"))

    else:
        with open(os.path.join(saved_calcs_path, "variances_across_ages_and_splits.pkl"), 'rb') as f:
            model_variance = pickle.load(f)
        with open(os.path.join(saved_calcs_path, "variance_stds_across_ages_and_splits.pkl"), 'rb') as f:
            model_variance_std = pickle.load(f)
        with open(os.path.join(saved_calcs_path, "model_data_cross_variances_across_splits.pkl"), 'rb') as f:
            model_data_cross_variances = pickle.load(f)
        with open(os.path.join(saved_calcs_path, "model_data_cross_variances_std_across_splits.pkl"), 'rb') as f:
            model_data_cross_variances_std = pickle.load(f)

    if do_save:
        with open(os.path.join(saved_calcs_path, "variances_across_ages_and_splits.pkl"), 'wb') as f:
            pickle.dump(model_variance, f)
        with open(os.path.join(saved_calcs_path, "variance_stds_across_ages_and_splits.pkl"), 'wb') as f:
            pickle.dump(model_variance_std, f)
        with open(os.path.join(saved_calcs_path, "model_data_cross_variances_across_splits.pkl"), 'wb') as f:
            pickle.dump(model_data_cross_variances, f)
        with open(os.path.join(saved_calcs_path, "model_data_cross_variances_std_across_splits.pkl"), 'wb') as f:
            pickle.dump(model_data_cross_variances_std, f)

    fig_size = RECT_LARGE_FIG_SIZE
    fontsize = FONT_SIZE
    markersize = MARKER_SIZE
    line_width = LINE_WIDTH
    main_axes = [0.17, 0.15, 0.8, 0.8]
    axes_labelpad = 1
    min_plotted_age = 300
    xticks = range(min_plotted_age, ADULT_WORM_AGE + 100, 800)
    yticks = np.arange(0.88, 1.01, 0.02)
    multiple_pruning_color = adjust_lightness('lightcoral', 1.0)

    for split in range(1, num_splits + 1):
        ages = np.arange(min_plotted_age, ADULT_WORM_AGE + 10, 10)
        model_variance_to_plot = model_variance[split - 1, min_plotted_age // 10 - 1:]
        model_variance_std_to_plot = model_variance_std[split - 1, min_plotted_age // 10 - 1:]

        fig1 = plt.figure(1, figsize=fig_size)
        ax1 = fig1.add_axes(main_axes)
        ax1.set_xticks(xticks)
        ax1.set_xticklabels([str(i) for i in xticks], fontsize=fontsize)
        ax1.set_yticks(yticks)
        ax1.set_yticklabels([f"{i:.2f}" for i in yticks], fontsize=fontsize)
        ax1.set_xlim(min_plotted_age - 50, ADULT_WORM_AGE + 100)
        ax1.set_ylim(0.88, 1.01)

        ax1.plot(ages, 1 - model_variance_to_plot, color=multiple_pruning_color, lw=line_width, label='model variance')
        plt.fill_between(ages, y1=np.clip(1 - model_variance_to_plot + model_variance_std_to_plot, a_min=None, a_max=1),
                         y2=np.clip(1 - model_variance_to_plot - model_variance_std_to_plot, a_min=0, a_max=None),
                         color=multiple_pruning_color, alpha=0.3)
        ax1.errorbar(data_ages, 1 - model_data_cross_variances[split - 1],
                     yerr=model_data_cross_variances_std[split - 1],
                     marker='o', ls='none', c='k', markersize=markersize * 1.5)
        ax1.set_xlabel('worm age [min.]', fontsize=fontsize, labelpad=axes_labelpad)
        ax1.set_ylabel('normalized Hamming similarity', fontsize=fontsize, labelpad=axes_labelpad)
        plt.savefig(os.path.join(out_path, f'model_variance_split{split}.pdf'), format='pdf')
        plt.show(block=False)
        plt.pause(3)
        plt.close()

    ages = np.arange(min_plotted_age, ADULT_WORM_AGE + 10, 10)
    model_variance_to_plot = model_variance[:, min_plotted_age // 10 - 1:].mean(axis=0)
    model_variance_std_to_plot = model_variance[:, min_plotted_age // 10 - 1:].std(axis=0)

    fig1 = plt.figure(1, figsize=fig_size)
    ax1 = fig1.add_axes(main_axes)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels([str(i) for i in xticks], fontsize=fontsize)
    ax1.set_yticks(yticks)
    ax1.set_yticklabels([f"{i:.2f}" for i in yticks], fontsize=fontsize)
    ax1.set_xlim(min_plotted_age - 50, ADULT_WORM_AGE + 100)
    ax1.set_ylim(0.88, 1.01)

    ax1.plot(ages, 1 - model_variance_to_plot, color=multiple_pruning_color, lw=line_width, label='model variance')
    plt.fill_between(ages, y1=np.clip(1 - model_variance_to_plot + model_variance_std_to_plot, a_min=None, a_max=1),
                     y2=1 - model_variance_to_plot - model_variance_std_to_plot,
                     color=multiple_pruning_color, alpha=0.3)
    ax1.errorbar(data_ages, 1 - model_data_cross_variances.mean(axis=0), yerr=model_data_cross_variances.std(axis=0),
                 marker='o', ls='none', c='k', markersize=markersize * 1.5,
                 label='model-train data cross variance training')
    ax1.set_xlabel('worm age [min.]', fontsize=fontsize, labelpad=axes_labelpad)
    ax1.set_ylabel('normalized Hamming similarity', fontsize=fontsize, labelpad=axes_labelpad)
    plt.savefig(os.path.join(out_path, f'average_model_variance_across_splits.pdf'), format='pdf')
    plt.show(block=False)
    plt.pause(3)
    plt.close()


def supplement_model_probs_hist(out_path=os.path.join("Figures", "FigS_multi_epochs_across_splits_7", "b")):
    with open("SavedOutputs\ReciprocalModel\\DyadsSplit\\max_likelihood_params_per_split_3_epochs.pkl",
              'rb') as f:
        max_like_params_multiple = pickle.load(f)
    num_splits = 20
    bin_res = 0.05
    num_bins = int(1 / bin_res)
    bins = np.arange(0, 1 + bin_res, bin_res)
    bin_middles = np.arange(bin_res / 2, 1, bin_res)
    norm_hists = np.zeros((num_splits, num_bins))

    main_axes = [0.17, 0.15, 0.8, 0.8]
    pylab.rcParams['xtick.major.pad'] = '0.5'
    pylab.rcParams['ytick.major.pad'] = '0.5'

    fontsize = FONT_SIZE
    axis_labelpad = 1
    xticks = np.arange(0, 1.2, 0.2)
    lowest_exp = -5
    yticks = [10 ** i for i in range(lowest_exp, 1)]
    minorticks = []
    for i in range(1, -lowest_exp + 1):
        minorticks += list(np.arange(2, 11) / 10 ** i)

    for split in range(1, num_splits + 1):
        smi = max_like_params_multiple[f'split{split}']['S-']
        model_test_dyads_path = f"SavedOutputs\ReciprocalModel\\DyadsSplit\\dyads_distributions\ThreeDevStages\TestSet\\split{split}\\{smi:.5f}\\3500.pkl"
        with open(model_test_dyads_path, 'rb') as f:
            model_test_dyads = pickle.load(f)
        model_probs = construct_probs_array_from_dyads_dist_string_keys(model_test_dyads)

        norm_hists[split - 1] = np.histogram(model_probs, bins=bins)[0] / model_probs.size

        fig = plt.figure(figsize=RECT_LARGE_FIG_SIZE)
        ax = fig.add_axes(main_axes)
        ax.set_yscale('log')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(10 ** lowest_exp, 1)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f'{tick:.1f}' for tick in xticks], fontsize=fontsize)
        ax.set_yticks(yticks)
        ax.set_yticklabels([r"$10^{{{0:d}}}$".format(i + lowest_exp) for i in range(len(yticks))], fontsize=fontsize)
        ax.yaxis.set_ticks(minorticks, minor=True)
        ax.set_xlabel('synaptic probability', fontsize=fontsize, labelpad=axis_labelpad)
        ax.set_ylabel('normalized frequency', fontsize=fontsize, labelpad=axis_labelpad)
        ax.bar(bin_middles, norm_hists[split - 1], align='center', width=bin_res, color='lightcoral')
        plt.savefig(os.path.join(out_path, f"model_probs_norm_hist_split{split}.pdf"), format='pdf')
        plt.show(block=False)
        plt.pause(3)
        plt.close()

    fig = plt.figure(figsize=RECT_LARGE_FIG_SIZE)
    ax = fig.add_axes(main_axes)
    ax.set_yscale('log')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(10 ** lowest_exp, 1)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'{tick:.1f}' for tick in xticks], fontsize=fontsize)
    ax.set_yticks(yticks)
    ax.set_yticklabels([r"$10^{{{0:d}}}$".format(i + lowest_exp) for i in range(len(yticks))], fontsize=fontsize)
    ax.yaxis.set_ticks(minorticks, minor=True)
    ax.set_xlabel('synaptic probability', fontsize=fontsize, labelpad=axis_labelpad)
    ax.set_ylabel('normalized frequency', fontsize=fontsize, labelpad=axis_labelpad)
    ax.bar(bin_middles, norm_hists.mean(axis=0), yerr=[np.zeros(num_bins), norm_hists.std(axis=0)], align='center',
           width=bin_res, color='lightcoral', zorder=0)
    plt.savefig(os.path.join(out_path, f"average_model_probs_norm_hist_across_splits.pdf"), format='pdf')
    plt.show(block=False)
    plt.pause(3)
    plt.close()


def prob_hist_by_num_datasets(out_path=os.path.join("Figures", "Fig7", "d")):
    worm_7_path = 'CElegansData\SubTypes\\connectomes\Dataset7.pkl'
    with open(worm_7_path, 'rb') as f:
        worm_7 = pickle.load(f)
    worm_8_path = 'CElegansData\SubTypes\\connectomes\Dataset8.pkl'
    with open(worm_8_path, 'rb') as f:
        worm_8 = pickle.load(f)
    worm_atlas_path = 'CElegansData\\worm_atlas_sub_connectome_chemical_no_autosynapses_subtypes.pkl'
    with open(worm_atlas_path, 'rb') as f:
        worm_atlas = pickle.load(f)

    data_sets_list = [worm_7, worm_8, worm_atlas]

    with open("SavedOutputs\ReciprocalModel\\DyadsSplit\\max_likelihood_params_per_split_3_epochs.pkl",
              'rb') as f:
        max_like_params_multiple = pickle.load(f)
    num_splits = 20
    train_or_test = "Test"

    bin_res = 0.01
    num_bins = int(1 / bin_res)
    bins = np.arange(0, 1 + bin_res, bin_res)
    bins_middles = np.arange(bin_res / 2, 1, bin_res)
    hists_not_exist = np.zeros((num_splits, num_bins))
    hists_exist_in_single_worm = np.zeros((num_splits, num_bins))
    hists_exist_in_2_worms = np.zeros((num_splits, num_bins))
    hists_exist_in_3_worms = np.zeros((num_splits, num_bins))
    num_high_prob_synapses = np.zeros(num_splits)

    fontsize = FONT_SIZE
    line_width = LINE_WIDTH / 2
    axis_labelpad = 1
    x_axis_ticks = np.arange(0, 1.2, 0.2)
    y_axis_ticks = np.arange(0, 0.226, 0.075)
    pylab.rcParams['xtick.major.pad'] = '0.5'
    pylab.rcParams['ytick.major.pad'] = '0.5'

    for split in range(1, num_splits + 1):
        smi = max_like_params_multiple[f'split{split}']['S-']
        model_test_dyads_dists_path = f"SavedOutputs\ReciprocalModel\\DyadsSplit\\dyads_distributions\ThreeDevStages\{train_or_test}Set\\split{split}\\{smi:.5f}\\3500.pkl"
        with open(model_test_dyads_dists_path, 'rb') as f:
            dyads_dists = pickle.load(f)

        probs_not_exist = []
        probs_exist_in_single_worm = []
        probs_exist_in_2_worms = []
        probs_exist_in_3_worms = []
        high_prob_synapses_num_exists = []
        for dyad in dyads_dists.keys():
            num_data_sets_where_exists_forth = 0
            num_data_sets_where_exists_back = 0
            for k in range(len(data_sets_list)):
                cur_data_set = data_sets_list[k]
                if dyad in cur_data_set.edges:
                    num_data_sets_where_exists_forth += 1
                if (dyad[1], dyad[0]) in cur_data_set.edges:
                    num_data_sets_where_exists_back += 1
            prob_forth = dyads_dists[dyad][RECIPROCAL_DYAD_IDX] + dyads_dists[dyad][ONLY_UPPER_TRIANGLE_SYNAPSE_IDX]
            prob_back = dyads_dists[dyad][RECIPROCAL_DYAD_IDX] + dyads_dists[dyad][ONLY_LOWER_TRIANGLE_SYNAPSE_IDX]
            if num_data_sets_where_exists_forth == 0:
                probs_not_exist.append(prob_forth)
            if num_data_sets_where_exists_back == 0:
                probs_not_exist.append(prob_back)
            if num_data_sets_where_exists_forth == 1:
                probs_exist_in_single_worm.append(prob_forth)
            if num_data_sets_where_exists_back == 1:
                probs_exist_in_single_worm.append(prob_back)
            if num_data_sets_where_exists_forth == 2:
                probs_exist_in_2_worms.append(prob_forth)
            if num_data_sets_where_exists_back == 2:
                probs_exist_in_2_worms.append(prob_back)
            if num_data_sets_where_exists_forth == 3:
                probs_exist_in_3_worms.append(prob_forth)
            if num_data_sets_where_exists_back == 3:
                probs_exist_in_3_worms.append(prob_back)

            if prob_forth > 0.95:
                high_prob_synapses_num_exists.append(num_data_sets_where_exists_forth)
            if prob_back > 0.95:
                high_prob_synapses_num_exists.append(num_data_sets_where_exists_back)

        hists_not_exist[split - 1] = np.histogram(probs_not_exist, bins=bins)[0] / len(probs_not_exist)
        hists_exist_in_single_worm[split - 1] = np.histogram(probs_exist_in_single_worm, bins=bins)[0] / len(
            probs_exist_in_single_worm)
        hists_exist_in_2_worms[split - 1] = np.histogram(probs_exist_in_2_worms, bins=bins)[0] / len(
            probs_exist_in_2_worms)
        hists_exist_in_3_worms[split - 1] = np.histogram(probs_exist_in_3_worms, bins=bins)[0] / len(
            probs_exist_in_3_worms)
        num_high_prob_synapses[split - 1] = len(high_prob_synapses_num_exists)

        fig = plt.figure(figsize=RECT_LARGE_FIG_SIZE)
        ax3 = fig.add_axes([0.17, 0.15, 0.8, 0.185])
        ax2 = fig.add_axes([0.17, 0.15 + 0.185 + 0.03, 0.8, 0.185])
        ax1 = fig.add_axes([0.17, 0.15 + 2 * (0.185 + 0.03), 0.8, 0.185])
        ax0 = fig.add_axes([0.17, 0.15 + 3 * (0.185 + 0.03), 0.8, 0.185])
        axes = [ax0, ax1, ax2, ax3]
        for ax in axes:
            ax.set_xlim(-0.01, 1.025)
            ax.set_ylim(-0.001, 0.215)
            ax.set_xticks(x_axis_ticks)
            ax.set_xticklabels(['' for tick in x_axis_ticks])
            ax.set_yticks(y_axis_ticks)
            ax.set_yticklabels([f'{y_axis_ticks[i]:.2f}' if i % 2 == 0 else '' for i in range(y_axis_ticks.size)],
                               fontsize=fontsize)
        ax0.set_ylabel('normalized frequency', fontsize=fontsize, labelpad=axis_labelpad,
                       y=-1.3)
        ax3.set_xticklabels([f'{tick:.1f}' for tick in x_axis_ticks], fontsize=fontsize)
        ax3.set_xlabel('synaptic probability', fontsize=fontsize, labelpad=axis_labelpad)

        ax0.bar(bins_middles, hists_not_exist[split - 1], align='center', width=bin_res,
                color='maroon')
        cur_mean = np.array(probs_not_exist).mean()
        ax0.plot([cur_mean, cur_mean], [0, 0.24], color='maroon', lw=line_width)
        ax1.bar(bins_middles, hists_exist_in_single_worm[split - 1], align='center', width=bin_res,
                color=adjust_lightness('darkorange', 0.75))
        cur_mean = np.array(probs_exist_in_single_worm).mean()
        ax1.plot([cur_mean, cur_mean], [0, 0.24], color=adjust_lightness('darkorange', 0.75), lw=line_width)
        ax2.bar(bins_middles, hists_exist_in_2_worms[split - 1], align='center', width=bin_res,
                color=adjust_lightness('gold', 0.9))
        cur_mean = np.array(probs_exist_in_2_worms).mean()
        ax2.plot([cur_mean, cur_mean], [0, 0.24], color=adjust_lightness('gold', 0.9), lw=line_width)
        ax3.bar(bins_middles, hists_exist_in_3_worms[split - 1], align='center', width=bin_res,
                color=adjust_lightness('green', 1.5))
        cur_mean = np.array(probs_exist_in_3_worms).mean()
        ax3.plot([cur_mean, cur_mean], [0, 0.24], color=adjust_lightness('green', 1.5), lw=line_width)
        plt.savefig(os.path.join(out_path, f'normalized_hists_by_synapse_group_split{split}.pdf'), format='pdf')
        plt.show(block=False)
        plt.pause(3)
        plt.close()

    fig = plt.figure(figsize=RECT_LARGE_FIG_SIZE)
    ax3 = fig.add_axes([0.17, 0.15, 0.8, 0.185])
    ax2 = fig.add_axes([0.17, 0.15 + 0.185 + 0.03, 0.8, 0.185])
    ax1 = fig.add_axes([0.17, 0.15 + 2 * (0.185 + 0.03), 0.8, 0.185])
    ax0 = fig.add_axes([0.17, 0.15 + 3 * (0.185 + 0.03), 0.8, 0.185])
    axes = [ax0, ax1, ax2, ax3]
    y_axis_ticks = np.arange(0, 0.31, 0.1)
    for ax in axes:
        ax.set_xlim(-0.01, 1.025)
        ax.set_ylim(-0.001, 0.3)
        ax.set_xticks(x_axis_ticks)
        ax.set_xticklabels(['' for tick in x_axis_ticks])
        ax.set_yticks(y_axis_ticks)
        ax.set_yticklabels([f'{y_axis_ticks[i]:.1f}' if i % 2 == 0 else '' for i in range(y_axis_ticks.size)],
                           fontsize=fontsize)
    ax0.set_ylabel('normalized frequency', fontsize=fontsize, labelpad=axis_labelpad,
                   y=-1.3)
    ax3.set_xticklabels([f'{tick:.1f}' for tick in x_axis_ticks], fontsize=fontsize)
    ax3.set_xlabel('synaptic probability', fontsize=fontsize, labelpad=axis_labelpad)

    ax0.bar(bins_middles, hists_not_exist.mean(axis=0),
            yerr=[np.zeros(bins_middles.shape), hists_not_exist.std(axis=0)], align='center', width=bin_res,
            color='maroon', error_kw=dict(lw=line_width))
    means_mean = (hists_not_exist.mean(axis=0) * bins_middles).sum()
    ax0.plot([means_mean, means_mean], [0, 0.3], color='maroon', lw=line_width)
    ax1.bar(bins_middles, hists_exist_in_single_worm.mean(axis=0),
            yerr=[np.zeros(bins_middles.shape), hists_exist_in_single_worm.std(axis=0)], align='center', width=bin_res,
            color=adjust_lightness('darkorange', 0.75), error_kw=dict(lw=line_width))
    means_mean = (hists_exist_in_single_worm.mean(axis=0) * bins_middles).sum()
    ax1.plot([means_mean, means_mean], [0, 0.3], color=adjust_lightness('darkorange', 0.75), lw=line_width)
    ax2.bar(bins_middles, hists_exist_in_2_worms.mean(axis=0),
            yerr=[np.zeros(bins_middles.shape), hists_exist_in_2_worms.std(axis=0)], align='center', width=bin_res,
            color=adjust_lightness('gold', 0.9), error_kw=dict(lw=line_width))
    means_mean = (hists_exist_in_2_worms.mean(axis=0) * bins_middles).sum()
    ax2.plot([means_mean, means_mean], [0, 0.3], color=adjust_lightness('gold', 0.9), lw=line_width)
    ax3.bar(bins_middles, hists_exist_in_3_worms.mean(axis=0),
            yerr=[np.zeros(bins_middles.shape), hists_exist_in_3_worms.std(axis=0)], align='center', width=bin_res,
            color=adjust_lightness('green', 1.5), error_kw=dict(lw=line_width))
    means_mean = (hists_exist_in_3_worms.mean(axis=0) * bins_middles).sum()
    ax3.plot([means_mean, means_mean], [0, 0.3], color=adjust_lightness('green', 1.5), lw=line_width)
    plt.savefig(os.path.join(out_path, f'average_normalized_hists_by_synapse_group_across_splits.pdf'), format='pdf')
    plt.show(block=False)
    plt.pause(3)
    plt.close()


def cum_prob_hist_by_num_datasets(out_path=os.path.join("Figures", "Fig7", "e")):
    worm_7_path = 'CElegansData\SubTypes\\connectomes\Dataset7.pkl'
    with open(worm_7_path, 'rb') as f:
        worm_7 = pickle.load(f)
    worm_8_path = 'CElegansData\SubTypes\\connectomes\Dataset8.pkl'
    with open(worm_8_path, 'rb') as f:
        worm_8 = pickle.load(f)
    worm_atlas_path = 'CElegansData\\worm_atlas_sub_connectome_chemical_no_autosynapses_subtypes.pkl'
    with open(worm_atlas_path, 'rb') as f:
        worm_atlas = pickle.load(f)

    data_sets_list = [worm_7, worm_8, worm_atlas]

    with open("SavedOutputs\ReciprocalModel\\DyadsSplit\\max_likelihood_params_per_split_3_epochs.pkl",
              'rb') as f:
        max_like_params_multiple = pickle.load(f)
    num_splits = 20

    bin_res = 0.001
    num_bins = int(1 / bin_res)
    bins = np.arange(0, 1 + bin_res, bin_res)
    bins_middles = np.arange(bin_res / 2, 1, bin_res)
    cumulative_not_exist = np.zeros((num_splits, num_bins))
    cumulative_exist_in_single_worm = np.zeros((num_splits, num_bins))
    cumulative_exist_in_2_worms = np.zeros((num_splits, num_bins))
    cumulative_exist_in_3_worms = np.zeros((num_splits, num_bins))

    fontsize = FONT_SIZE
    markersize = MARKER_SIZE
    line_width = LINE_WIDTH
    axis_labelpad = 1
    axes_ticks = np.arange(0, 1.2, 0.2)
    main_axes = [0.17, 0.15, 0.8, 0.8]
    pylab.rcParams['xtick.major.pad'] = '0.5'
    pylab.rcParams['ytick.major.pad'] = '0.5'

    for split in range(1, num_splits + 1):
        smi = max_like_params_multiple[f'split{split}']['S-']
        model_test_dyads_dists_path = f"SavedOutputs\\ReciprocalModel\\DyadsSplit\\dyads_distributions\ThreeDevStages\TestSet\\split{split}\\{smi:.5f}\\3500.pkl"
        with open(model_test_dyads_dists_path, 'rb') as f:
            dyads_dists = pickle.load(f)

        probs_not_exist = []
        probs_exist_in_single_worm = []
        probs_exist_in_2_worms = []
        probs_exist_in_3_worms = []
        for dyad in dyads_dists.keys():
            num_data_sets_where_exists_forth = 0
            num_data_sets_where_exists_back = 0
            for k in range(len(data_sets_list)):
                cur_data_set = data_sets_list[k]
                if dyad in cur_data_set.edges:
                    num_data_sets_where_exists_forth += 1
                if (dyad[1], dyad[0]) in cur_data_set.edges:
                    num_data_sets_where_exists_back += 1
            prob_forth = dyads_dists[dyad][RECIPROCAL_DYAD_IDX] + dyads_dists[dyad][ONLY_UPPER_TRIANGLE_SYNAPSE_IDX]
            prob_back = dyads_dists[dyad][RECIPROCAL_DYAD_IDX] + dyads_dists[dyad][ONLY_LOWER_TRIANGLE_SYNAPSE_IDX]
            if num_data_sets_where_exists_forth == 0:
                probs_not_exist.append(prob_forth)
            if num_data_sets_where_exists_back == 0:
                probs_not_exist.append(prob_back)
            if num_data_sets_where_exists_forth == 1:
                probs_exist_in_single_worm.append(prob_forth)
            if num_data_sets_where_exists_back == 1:
                probs_exist_in_single_worm.append(prob_back)
            if num_data_sets_where_exists_forth == 2:
                probs_exist_in_2_worms.append(prob_forth)
            if num_data_sets_where_exists_back == 2:
                probs_exist_in_2_worms.append(prob_back)
            if num_data_sets_where_exists_forth == 3:
                probs_exist_in_3_worms.append(prob_forth)
            if num_data_sets_where_exists_back == 3:
                probs_exist_in_3_worms.append(prob_back)

        fig = plt.figure(figsize=RECT_LARGE_FIG_SIZE)
        ax = fig.add_axes(main_axes)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(axes_ticks)
        ax.set_xticklabels([f'{tick:.1f}' for tick in axes_ticks], fontsize=fontsize)
        ax.set_yticks(axes_ticks)
        ax.set_yticklabels([f'{tick:.1f}' for tick in axes_ticks], fontsize=fontsize)
        ax.set_xlabel('synaptic probability', fontsize=fontsize, labelpad=axis_labelpad)
        ax.set_ylabel('normalized cumulative frequency', fontsize=fontsize, labelpad=axis_labelpad)
        cur_hist = np.histogram(probs_not_exist, bins=bins)[0]
        cumulative_not_exist[split - 1] = np.cumsum(cur_hist) / len(probs_not_exist)
        ax.plot(bins_middles, cumulative_not_exist[split - 1], c='maroon', markersize=markersize, lw=line_width,
                label='exist in no data set')
        cur_hist = np.histogram(probs_exist_in_single_worm, bins=bins)[0]
        cumulative_exist_in_single_worm[split - 1] = np.cumsum(cur_hist) / len(probs_exist_in_single_worm)
        ax.plot(bins_middles, cumulative_exist_in_single_worm[split - 1], c=adjust_lightness('darkorange', 0.75),
                markersize=markersize,
                lw=line_width,
                label='exist in one data set')
        cur_hist = np.histogram(probs_exist_in_2_worms, bins=bins)[0]
        cumulative_exist_in_2_worms[split - 1] = np.cumsum(cur_hist) / len(probs_exist_in_2_worms)
        ax.plot(bins_middles, cumulative_exist_in_2_worms[split - 1], c=adjust_lightness('gold', 0.9),
                markersize=markersize,
                lw=line_width, label='exist in 2 data sets')
        cur_hist = np.histogram(probs_exist_in_3_worms, bins=bins)[0]
        cumulative_exist_in_3_worms[split - 1] = np.cumsum(cur_hist) / len(probs_exist_in_3_worms)
        ax.plot(bins_middles, cumulative_exist_in_3_worms[split - 1], c=adjust_lightness('green', 1.5),
                markersize=markersize, lw=line_width,
                label='exist in 3 data sets')

        plt.savefig(os.path.join(out_path, f'norm_cum_dists_split{split}.pdf'), format='pdf')
        plt.show(block=False)
        plt.pause(3)
        plt.close()

    fig = plt.figure(figsize=RECT_LARGE_FIG_SIZE)
    ax = fig.add_axes(main_axes)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(axes_ticks)
    ax.set_xticklabels([f'{tick:.1f}' for tick in axes_ticks], fontsize=fontsize)
    ax.set_yticks(axes_ticks)
    ax.set_yticklabels([f'{tick:.1f}' for tick in axes_ticks], fontsize=fontsize)
    ax.set_xlabel('synaptic probability', fontsize=fontsize, labelpad=axis_labelpad)
    ax.set_ylabel('normalized cumulative frequency', fontsize=fontsize, labelpad=axis_labelpad)
    ax.plot(bins_middles, cumulative_not_exist.mean(axis=0), c='maroon', markersize=markersize, lw=line_width,
            label='exist in no data set')
    ax.fill_between(bins_middles, y1=cumulative_not_exist.mean(axis=0) - cumulative_not_exist.std(axis=0),
                    y2=cumulative_not_exist.mean(axis=0) + cumulative_not_exist.std(axis=0), color='maroon', alpha=0.3)
    ax.plot(bins_middles, cumulative_exist_in_single_worm.mean(axis=0), color=adjust_lightness('darkorange', 0.75),
            markersize=markersize,
            lw=line_width,
            label='exist in one data set')
    ax.fill_between(bins_middles,
                    y1=cumulative_exist_in_single_worm.mean(axis=0) - cumulative_exist_in_single_worm.std(axis=0),
                    y2=cumulative_exist_in_single_worm.mean(axis=0) + cumulative_exist_in_single_worm.std(axis=0),
                    color='darkorange', alpha=0.3)
    ax.plot(bins_middles, cumulative_exist_in_2_worms.mean(axis=0), c=adjust_lightness('gold', 0.9),
            markersize=markersize,
            lw=line_width, label='exist in 2 data sets')
    ax.fill_between(bins_middles,
                    y1=cumulative_exist_in_2_worms.mean(axis=0) - cumulative_exist_in_2_worms.std(axis=0),
                    y2=cumulative_exist_in_2_worms.mean(axis=0) + cumulative_exist_in_2_worms.std(axis=0),
                    color=adjust_lightness('gold', 0.9), alpha=0.3)
    ax.plot(bins_middles, cumulative_exist_in_3_worms.mean(axis=0), c=adjust_lightness('green', 1.5),
            markersize=markersize, lw=line_width,
            label='exist in 3 data sets')
    ax.fill_between(bins_middles,
                    y1=cumulative_exist_in_3_worms.mean(axis=0) - cumulative_exist_in_3_worms.std(axis=0),
                    y2=cumulative_exist_in_3_worms.mean(axis=0) + cumulative_exist_in_3_worms.std(axis=0),
                    color=adjust_lightness('green', 1.5), alpha=0.3)

    plt.savefig(os.path.join(out_path, f'average_norm_cum_dists_split_across_splits.pdf'), format='pdf')
    plt.show(block=False)
    plt.pause(3)
    plt.close()


def _get_mean_weighted_conenctome():
    worm_7_connectivity_df = pd.read_excel(os.path.join("CElegansData", "synapse_count_matrices.xlsx"),
                                           sheet_name='Dataset7', index_col=2, skiprows=2)
    worm_7_connectivity_df = worm_7_connectivity_df[1:]
    worm_7_connectivity_df = worm_7_connectivity_df.iloc[:, 2:]

    worm_8_connectivity_df = pd.read_excel(os.path.join("CElegansData", "synapse_count_matrices.xlsx"),
                                           sheet_name='Dataset8', index_col=2, skiprows=2)
    worm_8_connectivity_df = worm_8_connectivity_df[1:]
    worm_8_connectivity_df = worm_8_connectivity_df.iloc[:, 2:]

    with open(os.path.join("CElegansData", "nerve_ring_neurons_subset.pkl"), 'rb') as f:
        nerve_ring_neurons = sorted(pickle.load(f))

    num_neurons = len(nerve_ring_neurons)
    worm_7_syn_count = np.zeros((num_neurons, num_neurons))
    worm_8_syn_count = np.zeros((num_neurons, num_neurons))
    for i, pre in enumerate(nerve_ring_neurons):
        for j, post in enumerate(nerve_ring_neurons):
            if pre == post:
                continue
            worm_7_syn_count[i, j] = worm_7_connectivity_df.loc[post, pre]
            worm_8_syn_count[i, j] = worm_8_connectivity_df.loc[post, pre]
    return (worm_7_syn_count + worm_8_syn_count) / 2


def mean_weighted_connectivity_matrix(
        out_path=os.path.join("Figures", "Fig7", "mean_weighted_connectivity_matrix.pdf")):
    mean_weighed_connectome = _get_mean_weighted_conenctome()
    neurons_idx_by_type = _get_neurons_idx_by_inferred_type(8)

    axis_ticks = range(0, 181, 60)
    fontsize = FONT_SIZE
    main_axes = [0.18, 0.18, 0.65, 0.65]
    colorbar_axes = [0.85, 0.18, 0.03, 0.65]
    pylab.rcParams['xtick.major.pad'] = '0.5'
    pylab.rcParams['ytick.major.pad'] = '0.5'
    axis_labelpad = 1
    colorbar_labelpad = -1
    dy = 0.5
    fig = plt.figure(figsize=SQUARE_FIG_SIZE)
    ax = fig.add_axes(main_axes)
    im = ax.imshow(mean_weighed_connectome[neurons_idx_by_type, neurons_idx_by_type.T], cmap="Grays")
    ax.set_xlabel("post-synaptic neuronal idx", fontsize=fontsize, labelpad=axis_labelpad)
    ax.set_ylabel("pre-synaptic neuronal idx", fontsize=fontsize, labelpad=axis_labelpad)
    ax.set_xticks(axis_ticks)
    ax.set_xticklabels([str(i) for i in axis_ticks], fontsize=fontsize)
    ax.set_yticks(axis_ticks)
    ax.set_yticklabels([str(i) for i in axis_ticks], fontsize=fontsize)
    cbar_ax = fig.add_axes(colorbar_axes)
    cbar3 = fig.colorbar(im, cax=cbar_ax, ticks=[0, 10, 20, 30])
    cbar_ax.set_yticklabels(['0', '', '', '30'], fontsize=fontsize)
    cbar3.set_label('# synapses', rotation=270, labelpad=colorbar_labelpad, y=dy, fontsize=fontsize)
    fig.savefig(out_path, format='pdf')
    plt.show()


def prediction_vs_num_synapses_data(out_path=os.path.join("Figures", "Fig7", "prediction_vs_num_synapses.pdf")):
    with open(os.path.join("CElegansData", "nerve_ring_neurons_subset.pkl"), 'rb') as f:
        nerve_ring_neurons = sorted(pickle.load(f))

    mean_weighted_connectome = _get_mean_weighted_conenctome()

    with open(os.path.join("SavedOutputs", "ReciprocalModel", "DyadsSplit",
                           "max_likelihood_params_per_split_3_epochs.pkl"), 'rb') as f:
        max_like_params_multiple = pickle.load(f)
    split_to_plot = 16
    smi = max_like_params_multiple[f'split{split_to_plot}']['S-']
    model_train_dyads_path = os.path.join("SavedOutputs", "ReciprocalModel", "DyadsSplit", "dyads_distributions",
                                          "ThreeDevStages", "TrainSet", f"split{split_to_plot}", f"{smi:.5f}",
                                          "3500.pkl")
    with open(model_train_dyads_path, 'rb') as f:
        model_train_dyads = pickle.load(f)
    model_test_dyads_path = os.path.join("SavedOutputs", "ReciprocalModel", "DyadsSplit", "dyads_distributions",
                                         "ThreeDevStages", "TestSet", f"split{split_to_plot}", f"{smi:.5f}", "3500.pkl")
    with open(model_test_dyads_path, 'rb') as f:
        model_test_dyads = pickle.load(f)
    model_av_mat = calc_reciprocal_dependence_model_average_adj_mat_from_dyads_distributions_str_keys(
        model_train_dyads, model_test_dyads, nerve_ring_neurons)

    num_synapses_to_prob_dict = {}
    for i, pre in enumerate(nerve_ring_neurons):
        for j, post in enumerate(nerve_ring_neurons):
            if pre == post:
                continue
            cur_num_synapses = mean_weighted_connectome[i, j]
            if cur_num_synapses not in num_synapses_to_prob_dict.keys():
                num_synapses_to_prob_dict[cur_num_synapses] = [model_av_mat[i, j]]
            else:
                num_synapses_to_prob_dict[cur_num_synapses].append(model_av_mat[i, j])

    existing_num_synapses = sorted(list(num_synapses_to_prob_dict.keys()))
    mean_probs = [np.array(num_synapses_to_prob_dict[i]).mean() for i in existing_num_synapses]
    std_probs = [np.array(num_synapses_to_prob_dict[i]).std() for i in existing_num_synapses]

    pylab.rcParams['xtick.major.pad'] = '0.5'
    pylab.rcParams['ytick.major.pad'] = '0.5'
    fig = plt.figure(figsize=(2 * SQUARE_FIG_SIZE[0], SQUARE_FIG_SIZE[1]))
    ax = fig.add_subplot([0.11, 0.18, 0.73, 0.78])
    xticks = [0, 20, 40]
    yticks = [0.0, 0.5, 1.0]
    markersize = MARKER_SIZE
    axis_labelpad = -1
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(i) for i in xticks], fontsize=FONT_SIZE)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{i:.1f}' for i in yticks], fontsize=FONT_SIZE)
    ax.set_xlim([-1, 41])
    ax.set_ylim([-0.06, 1.06])
    ax.set_xlabel('# synapses', fontsize=FONT_SIZE, labelpad=axis_labelpad)
    ax.set_ylabel('model prediction', fontsize=FONT_SIZE, labelpad=axis_labelpad)

    num_synapses_freqs = [len(num_synapses_to_prob_dict[i]) for i in existing_num_synapses]
    norm = colors.LogNorm(vmin=0.1 * min(num_synapses_freqs), vmax=max(num_synapses_freqs))
    cs = plt.cm.gray_r(norm(num_synapses_freqs))
    for xi, yi, yerri, ci in zip(existing_num_synapses, mean_probs, std_probs, cs):
        ax.errorbar(xi, yi, yerr=yerri, fmt='o', color=ci, ecolor=ci, markersize=markersize / 2, elinewidth=0.75)

    sm = plt.cm.ScalarMappable(cmap='gray_r', norm=norm)
    sm.set_array([])  # Dummy array for colorbar scaling
    cbar_ax = fig.add_axes([0.85, 0.18, 0.02, 0.78])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar_majorticks = np.logspace(0, 4, num=5)
    cbar_ax.set_yticks(cbar_majorticks)
    cbar_minorticks = []
    for i in range(5):
        cbar_minorticks += list(np.arange(2, 11) * 10 ** i)
    cbar_ax.set_yticks(cbar_minorticks, minor=True)
    cbar_ax.set_yticklabels([r"$10^{{{0:d}}}$".format(i) if i % 2 == 0 else '' for i in range(len(cbar_majorticks))],
                            fontsize=FONT_SIZE)
    cbar_ax.set_ylim(0, max(num_synapses_freqs))
    cbar.set_label('# neuronal pairs', fontsize=FONT_SIZE)

    fig.savefig(out_path, format="pdf")
    plt.show()


def supplement_noisy_birth_times(out_path="Figures\\FigS_noised_birth_times"):
    data_connectome_path = 'CElegansData\SubTypes\\connectomes\Dataset8.pkl'
    with open(data_connectome_path, 'rb') as f:
        data_connectome = pickle.load(f)
    data_adj_mat = nx.to_numpy_array(data_connectome, nodelist=sorted(data_connectome.nodes))
    data_adj_mat = data_adj_mat.astype(int)

    likelihoods_path = 'SavedOutputs\IndependentModel\likelihoods\SubTypes'
    smi, beta, _ = find_max_likelihood_full_model(likelihoods_path)
    model_average_adj_mat_path = f"SavedOutputs\IndependentModel\\average_adj_mats\SubTypes\\smi{smi:.5f}_beta{beta:.5f}_adult.pkl"
    with open(model_average_adj_mat_path, 'rb') as f:
        true_birth_times_average_adj_mat = pickle.load(f)

    true_birth_times_auc = roc_auc_score(data_adj_mat.flatten(), true_birth_times_average_adj_mat.flatten())
    true_birth_times_like = average_matrix_log_likelihood(true_birth_times_average_adj_mat, data_connectome_path)

    noisy_birth_times_average_connectomes_path = "SavedOutputs\IndependentModel\\average_adj_mats\SubTypes\\noised_birth_times"
    noise_range = np.arange(0.1, 1.1, 0.1)
    num_noisings = len(
        os.listdir(os.path.join(noisy_birth_times_average_connectomes_path, f'{int(100 * noise_range[0])}%_noise')))
    noisy_birth_times_likes = np.zeros((num_noisings, noise_range.size))
    noisy_birth_times_aucs = np.zeros((num_noisings, noise_range.size))
    for noising in tqdm(range(1, num_noisings + 1)):
        noise_idx = 0
        for noise_level in noise_range:
            model_noisy_birth_times_path = os.path.join(noisy_birth_times_average_connectomes_path,
                                                        f'{int(100 * noise_level)}%_noise', f'{noising}.pkl')
            with open(model_noisy_birth_times_path, 'rb') as f:
                model_noisy_average_mat = pickle.load(f)
            noisy_birth_times_likes[noising - 1, noise_idx] = average_matrix_log_likelihood(model_noisy_average_mat,
                                                                                            data_connectome_path)
            noisy_birth_times_aucs[noising - 1, noise_idx] = roc_auc_score(data_adj_mat.flatten(),
                                                                           model_noisy_average_mat.flatten())
            noise_idx += 1

    fig = plt.figure(figsize=RECT_LARGE_FIG_SIZE)
    main_axes = [0.21, 0.17, 0.76, 0.8]
    fontsize = FONT_SIZE
    line_width = LINE_WIDTH
    markersize = MARKER_SIZE
    axes_labelpad = 1
    xticks = np.arange(0.1, 1.2, 0.3)
    yticks = range(-10, 11, 5)
    ax = fig.add_axes(main_axes)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'{int(100 * tick)}%' for tick in xticks], fontsize=fontsize)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{tick}' for tick in yticks], fontsize=fontsize)
    ax.set_xlabel('noise level', fontsize=fontsize, labelpad=axes_labelpad)
    ax.set_ylabel(r'$\log _{10} ( \mathcal{L} _{\mathrm{true}} / \mathcal{L} _{\mathrm{noisy}} )$', fontsize=fontsize,
                  labelpad=axes_labelpad)
    ax.set_xlim(0, 1.1)
    ax.set_ylim(-11, 11)
    ax.plot([noise_range[0] - 0.05, noise_range[-1] + 0.05], [0, 0],
            c=adjust_lightness('grey', 1.5), markersize=markersize)
    ax.errorbar(noise_range, np.log10(np.exp(1)) * (true_birth_times_like - noisy_birth_times_likes).mean(axis=0),
                yerr=(np.log10(np.exp(1)) * (true_birth_times_like - noisy_birth_times_likes)).std(axis=0), marker='.',
                c='k', lw=line_width, ms=3 * markersize)
    plt.savefig(os.path.join(out_path, "true_noisy_log_like_ratio_subtypes.pdf"), format='pdf')
    plt.show()

    fig = plt.figure(figsize=RECT_LARGE_FIG_SIZE)
    yticks = np.arange(-0.01, 0.011, 0.005)
    ax = fig.add_axes(main_axes)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'{int(100 * tick)}%' for tick in xticks], fontsize=fontsize)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{tick:.3f}' for tick in yticks], fontsize=fontsize)
    ax.set_xlabel('noise level', fontsize=fontsize, labelpad=axes_labelpad)
    ax.set_ylabel(r'$\Delta \mathrm{AUC}$', fontsize=fontsize, labelpad=axes_labelpad)
    ax.set_xlim(0, 1.1)
    ax.set_ylim(-0.01, 0.01)
    ax.plot([noise_range[0] - 0.05, noise_range[-1] + 0.05], [0, 0],
            c=adjust_lightness('grey', 1.5), markersize=markersize)
    ax.errorbar(noise_range, (true_birth_times_auc - noisy_birth_times_aucs).mean(axis=0),
                yerr=(true_birth_times_auc - noisy_birth_times_aucs).std(axis=0), marker='.',
                c='k', lw=line_width, ms=3 * markersize)
    plt.savefig(os.path.join(out_path, "true_noisy_auc_diff_subtypes.pdf"), format='pdf')
    plt.show()


def supplement_auc_vs_number_of_inferred_types(out_path="Figures\\FigS_num_types_choice"):
    data_connectome_path = 'CElegansData\SubTypes\\connectomes\Dataset8.pkl'
    with open(data_connectome_path, 'rb') as f:
        data_connectome = pickle.load(f)
    data_adj_mat = nx.to_numpy_array(data_connectome, nodelist=sorted(data_connectome.nodes))
    data_adj_mat = data_adj_mat.astype(int)
    average_connectomes_dir_path = "SavedOutputs\IndependentModel\\average_adj_mats\InferredTypes"
    possible_num_types = np.arange(1, 51, 1)
    aucs = np.zeros(possible_num_types.size)
    for num_types in possible_num_types:
        cur_file_name = os.path.join(average_connectomes_dir_path, f"{int(num_types)}_types.pkl")
        with open(cur_file_name, 'rb') as f:
            cur_average_mat = pickle.load(f)
        aucs[num_types - 1] = roc_auc_score(data_adj_mat.flatten(), cur_average_mat.flatten())
    fig = plt.figure(figsize=RECT_MEDIUM_FIG_SIZE)
    ax1 = fig.add_axes([0.18, 0.16, 0.8, 0.8])
    fontsize = FONT_SIZE
    line_width = LINE_WIDTH
    markersize = MARKER_SIZE * 3
    axis_labelpad = 1
    x_axes_ticks = np.arange(0, 51, 10)
    y_axes_ticks = np.arange(0.5, 1, 0.1)
    ax1.set_xticks(x_axes_ticks)
    ax1.set_xticklabels([f'{tick}' for tick in x_axes_ticks], fontsize=fontsize)
    ax1.set_yticks(y_axes_ticks)
    ax1.set_yticklabels([f'{tick:.1f}' for tick in y_axes_ticks], fontsize=fontsize)
    ax1.set_xlabel('number of neuronal cell types', fontsize=fontsize, labelpad=axis_labelpad)
    ax1.set_ylabel('AUROC', fontsize=fontsize, labelpad=axis_labelpad)
    ax1.set_xlim(0, 51)
    ax1.set_ylim(0.5, 0.9)
    ax1.plot(possible_num_types, aucs, marker='.',
             c=SINGLE_EPOCH_INFERRED_COLOR, lw=line_width,
             markersize=markersize, label='full model')
    plt.savefig(os.path.join(out_path, "auc_vs_num_types.pdf"), format='pdf')
    plt.show()


def supplement_outputs_control_overfit(out_path="Figures\\FigS_num_types_choice"):
    max_num_learned_types = 15
    num_control_types = 8
    num_worm_samples_for_control_test = 100
    aucs_data = np.zeros(max_num_learned_types)
    aucs_controls_means_test = np.zeros(max_num_learned_types)
    aucs_controls_stds_test = np.zeros(max_num_learned_types)
    with open("CElegansData\SubTypes\\connectomes\Dataset8.pkl", 'rb') as f:
        data_connectome = pickle.load(f)
        data_adj_mat = nx.to_numpy_array(data_connectome, nodelist=sorted(data_connectome.nodes))
        data_adj_mat = data_adj_mat.flatten()
        data_adj_mat = data_adj_mat.astype(int)

    average_connectomes_data_trained_path = "SavedOutputs\IndependentModel\\average_adj_mats\InferredTypes"
    average_connectomes_control_path = f"SavedOutputs\ModelOutputsControl\\average_adj_mats\\{num_control_types}_control_types"
    data_based_model_average_mat_path = os.path.join(average_connectomes_data_trained_path,
                                                     f"{num_control_types}_types.pkl")
    with open(data_based_model_average_mat_path, 'rb') as f:
        data_based_model_average_mat = pickle.load(f)
    for number in range(1, max_num_learned_types + 1):
        with open(os.path.join(average_connectomes_data_trained_path, f'{number}_types.pkl'), 'rb') as f:
            model_av_adj_mat = pickle.load(f).flatten()
        aucs_data[number - 1] = roc_auc_score(data_adj_mat, model_av_adj_mat)

        with open(os.path.join(average_connectomes_control_path, f'{number}_learned_types.pkl'), 'rb') as f:
            control_average_mat = pickle.load(f)
        aucs = np.zeros(num_worm_samples_for_control_test)
        for i in range(num_worm_samples_for_control_test):
            worm_i = sample_from_average_adj_mat(data_based_model_average_mat)
            aucs[i] = roc_auc_score(worm_i.flatten(), control_average_mat.flatten())

        aucs_controls_means_test[number - 1] = aucs.mean()
        aucs_controls_stds_test[number - 1] = aucs.std()

    fig = plt.figure(figsize=RECT_MEDIUM_FIG_SIZE)
    ax1 = fig.add_axes([0.18, 0.16, 0.8, 0.8])
    fontsize = FONT_SIZE
    line_width = LINE_WIDTH
    markersize = MARKER_SIZE * 3
    axis_labelpad = 1
    x_axes_ticks = np.arange(0, max_num_learned_types + 1, 5)
    y_axes_ticks = np.arange(0.5, 1, 0.1)
    ax1.set_xticks(x_axes_ticks)
    ax1.set_xticklabels([f'{tick}' for tick in x_axes_ticks], fontsize=fontsize)
    ax1.set_yticks(y_axes_ticks)
    ax1.set_yticklabels([f'{tick:.1f}' for tick in y_axes_ticks], fontsize=fontsize)
    ax1.set_xlabel('number of neuronal cell types', fontsize=fontsize, labelpad=axis_labelpad)
    ax1.set_ylabel('AUROC', fontsize=fontsize, labelpad=axis_labelpad)
    ax1.set_xlim(0, max_num_learned_types + 1)
    ax1.set_ylim(0.5, 0.9)
    ax1.errorbar(range(1, max_num_learned_types + 1), aucs_controls_means_test, yerr=aucs_controls_stds_test,
                 marker='.',
                 c='k', lw=line_width,
                 markersize=markersize)
    ax1.plot(range(1, max_num_learned_types + 1), aucs_data, marker='.',
             c=SINGLE_EPOCH_INFERRED_COLOR, lw=line_width,
             markersize=markersize)
    plt.savefig(os.path.join(out_path, "auc_vs_num_types_model_outputs.pdf"), format='pdf')
    plt.show()


def fig_compressed_models_a_b(out_path="Figures\\FigCompressedModels"):
    num_types = 8
    smi, beta, _ = find_max_likelihood_full_model(
        f"SavedOutputs\IndependentModel\likelihoods\InferredTypes\\{num_types}_types")
    spls_path = os.path.join(
        f"SavedOutputs\IndependentModel\S+s\InferredTypes\\{num_types}_types",
        f"spls_smi{smi:.5f}_beta{beta:.5f}.pkl")

    sorted_inferred_types_indices = np.array([4, 6, 7, 5, 1, 0, 3, 2]).reshape(num_types, 1)
    spls_mat, _ = convert_spls_dict_to_mat(spls_path, developmental_stage=0)
    spls_mat = spls_mat[sorted_inferred_types_indices, sorted_inferred_types_indices.T]

    compact_spls_path = f"SavedOutputs\IndependentModel\compact_models\\0.05_performance_decrease\\{num_types}\\compact_spls.pkl"
    compact_spls_mat, _ = convert_spls_dict_to_mat(compact_spls_path, developmental_stage=0)
    compact_spls_mat = compact_spls_mat[sorted_inferred_types_indices, sorted_inferred_types_indices.T]

    max_value = np.log10(spls_mat.max())
    min_value = -5
    fontsize = FONT_SIZE
    main_axes = [0.16, 0.18, 0.65, 0.65]
    colorbar_axes = [0.82, 0.18, 0.025, 0.65]
    pylab.rcParams['xtick.major.pad'] = '0.5'
    pylab.rcParams['ytick.major.pad'] = '0.5'
    axis_labelpad = 1
    colorbar_labelpad = 3
    fig = plt.figure(figsize=SQUARE_FIG_SIZE)
    ax = fig.add_axes(main_axes)
    im = ax.imshow(spls_mat, cmap="Greens", norm=colors.LogNorm(vmin=10 ** min_value, vmax=10 ** max_value))
    cbar_ax = fig.add_axes(colorbar_axes)
    majorticks = np.logspace(min_value, max_value, num=6)
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=majorticks)
    minorticks = []
    for i in range(1, 6):
        minorticks += list(np.arange(2, 11) / 10 ** i)
    cbar.ax.yaxis.set_ticks(minorticks, minor=True)
    cbar_ax.set_yticklabels([r"$10^{{{0:d}}}$".format(i - len(majorticks) + 1) for i in range(len(majorticks))],
                            fontsize=fontsize)
    cbar.ax.set_title('$S^{+}$', fontsize=fontsize, pad=colorbar_labelpad, loc='left')
    ax.set_ylabel('pre-synaptic neuronal type', fontsize=fontsize, labelpad=axis_labelpad)
    ax.set_xlabel('post-synaptic neuronal type', fontsize=fontsize, labelpad=axis_labelpad)
    ax.set_xticks(range(num_types))
    ax.set_xticklabels([f'C{idx}' for idx in range(num_types)], fontsize=fontsize)
    ax.set_yticks(range(num_types))
    ax.set_yticklabels([f'C{idx}' for idx in range(num_types)], fontsize=fontsize)
    plt.savefig(os.path.join(out_path, 'S+_full.pdf'), format='pdf')
    plt.show()

    fig = plt.figure(figsize=SQUARE_FIG_SIZE)
    ax = fig.add_axes(main_axes)
    im = ax.imshow(compact_spls_mat, cmap="Greens", norm=colors.LogNorm(vmin=10 ** min_value, vmax=10 ** max_value))
    cbar_ax = fig.add_axes(colorbar_axes)
    majorticks = np.logspace(min_value, max_value, num=6)
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=majorticks)
    minorticks = []
    for i in range(1, 6):
        minorticks += list(np.arange(2, 11) / 10 ** i)
    cbar.ax.yaxis.set_ticks(minorticks, minor=True)
    cbar_ax.set_yticklabels([r"$10^{{{0:d}}}$".format(i - len(majorticks) + 1) for i in range(len(majorticks))],
                            fontsize=fontsize)
    cbar.ax.set_title('$S^{+}$', fontsize=fontsize, pad=colorbar_labelpad, loc='left')
    ax.set_ylabel('pre-synaptic neuronal type', fontsize=fontsize, labelpad=axis_labelpad)
    ax.set_xlabel('post-synaptic neuronal type', fontsize=fontsize, labelpad=axis_labelpad)
    ax.set_xticks(range(num_types))
    ax.set_xticklabels([f'C{idx}' for idx in range(num_types)], fontsize=fontsize)
    ax.set_yticks(range(num_types))
    ax.set_yticklabels([f'C{idx}' for idx in range(num_types)], fontsize=fontsize)
    plt.savefig(os.path.join(out_path, 'S+_quantized.pdf'), format='pdf')
    plt.show()


def fig_compressed_models_c(out_path="Figures\\FigCompressedModels"):
    num_types = 8
    data_connectome_path = 'CElegansData\SubTypes\\connectomes\Dataset8.pkl'
    with open(data_connectome_path, 'rb') as f:
        data_connectome = pickle.load(f)
    data_adj_mat = nx.to_numpy_array(data_connectome, nodelist=sorted(data_connectome.nodes))
    data_adj_mat = data_adj_mat.astype(int)

    compressed_models_num_spls = np.arange(2, 52)
    compressed_models_path = os.path.join("SavedOutputs", "IndependentModel", "compact_models",
                                          f"{num_types}_types_model_compressions")
    models_auc = np.zeros(compressed_models_num_spls.size + 1)
    for i, compressed_size in enumerate(compressed_models_num_spls):
        with open(os.path.join(compressed_models_path, f'{compressed_size}_spl_params', 'auc.pkl'), 'rb') as f:
            models_auc[i] = pickle.load(f)

    full_model_av_mat_path = os.path.join("SavedOutputs", "IndependentModel", "average_adj_mats", "InferredTypes",
                                          f'{num_types}_types.pkl')
    with open(full_model_av_mat_path, 'rb') as f:
        full_model_av_mat = pickle.load(f)
    models_auc[-1] = roc_auc_score(data_adj_mat.flatten(), full_model_av_mat.flatten())

    fontsize = FONT_SIZE
    markersize = 3 * MARKER_SIZE
    line_width = LINE_WIDTH
    axes_labelpad = 1
    pylab.rcParams['ytick.major.pad'] = '0.5'
    fig = plt.figure(figsize=SQUARE_FIG_SIZE)
    ax1 = fig.add_axes([0.2, 0.18, 0.75, 0.75])
    ax1.set_xlim(0, 70)
    ax1.set_ylim(0.5, 0.85)
    x_axes_ticks = np.arange(5, 66, 20)
    y_axes_ticks = np.arange(0.5, 0.9, 0.1)
    ax1.set_xticks(x_axes_ticks)
    ax1.set_xticklabels([f'{tick}' for tick in x_axes_ticks], fontsize=fontsize)
    ax1.set_yticks(y_axes_ticks)
    ax1.set_yticklabels([f'{tick:.1f}' for tick in y_axes_ticks], fontsize=fontsize)
    ax1.set_xlabel('# model parameters', fontsize=fontsize, labelpad=axes_labelpad)
    ax1.set_ylabel('AUROC', fontsize=fontsize, labelpad=axes_labelpad)

    models_num_parameters = np.zeros(compressed_models_num_spls.size + 1)
    models_num_parameters[:-1] = compressed_models_num_spls
    models_num_parameters[-1] = num_types ** 2
    models_num_parameters += 2
    ax1.scatter(models_num_parameters, models_auc, color='k', s=markersize, zorder=2)
    ax1.scatter([models_num_parameters[5], models_num_parameters[-1]], [models_auc[5], models_auc[-1]],
                color=[0.5 * np.ones(3), SINGLE_EPOCH_INFERRED_COLOR], s=markersize, zorder=3)
    ax1.plot(models_num_parameters, models_auc, color='k', lw=line_width, zorder=1)
    plt.savefig(os.path.join(out_path, "auc_vs_num_params.pdf"), format='pdf')
    plt.show()


def fig_compressed_models_d(out_path="Figures\\FigCompressedModels"):
    num_types_range = range(1, 16)
    compact_num_params_5_per_performance = np.zeros(len(num_types_range))
    compact_models_path = os.path.join("SavedOutputs", "IndependentModel", "compact_models")
    for num_types in num_types_range:
        with open(os.path.join(compact_models_path, f"0.05_performance_decrease", f"{num_types}", "compact_size.pkl"),
                  'rb') as f:
            compact_num_params_5_per_performance[num_types - min(num_types_range)] = pickle.load(f) + 2

    fontsize = FONT_SIZE
    markersize = 3 * MARKER_SIZE
    line_width = LINE_WIDTH
    axes_labelpad = 1
    pylab.rcParams['ytick.major.pad'] = '0.5'
    fig = plt.figure(figsize=SQUARE_FIG_SIZE)
    ax1 = fig.add_axes([0.2, 0.18, 0.75, 0.75])
    ax1.set_xlim(0, 16)
    ax1.set_ylim(-5, 235)
    x_axes_ticks = np.arange(0, 16, 5)
    y_axes_ticks = np.arange(0, 235, 50)
    ax1.set_xticks(x_axes_ticks)
    ax1.set_xticklabels([f'{tick}' for tick in x_axes_ticks], fontsize=fontsize)
    ax1.set_yticks(y_axes_ticks)
    ax1.tick_params(axis='y', labelsize=fontsize)
    ax1.set_xlabel('# neuronal types', fontsize=fontsize, labelpad=axes_labelpad)
    ax1.set_ylabel('# model parameters', fontsize=fontsize, labelpad=axes_labelpad)

    ax1.plot(num_types_range, compact_num_params_5_per_performance, color=np.ones(3) * 0.5, marker='.', lw=line_width,
             markersize=markersize, label='5% performance loss')
    ax1.plot(num_types_range, np.array(num_types_range) ** 2 + 2, color=SINGLE_EPOCH_INFERRED_COLOR, marker='.',
             lw=line_width, markersize=markersize)
    plt.savefig(os.path.join(out_path, "num_types_in_compact_models.pdf"), format='pdf')
    plt.show()


def supplement_reciprocity_independent_model(out_path="Figures\\FigS_reciprocity"):
    max_type_number = 50
    mean_reciprocities = np.zeros(max_type_number)
    std_reciprocities = np.zeros(max_type_number)
    with open("CElegansData\SubTypes\\connectomes\Dataset7.pkl", 'rb') as f:
        data_connectome = pickle.load(f)
    data_reciprocity = calc_reciprocity(data_connectome)
    num_neurons = len(data_connectome.nodes)
    num_synapses = num_neurons * (num_neurons - 1)
    num_pairs_of_neurons = num_synapses / 2
    for number in range(1, mean_reciprocities.size + 1):
        with open(f"SavedOutputs\IndependentModel\\average_adj_mats\InferredTypes\\{number}_types.pkl", 'rb') as f:
            model_av_adj_mat = pickle.load(f)

        reciprocity_prob_matrix = np.triu(model_av_adj_mat * model_av_adj_mat.T)
        mean_reciprocities[number - 1] = reciprocity_prob_matrix.sum() / num_pairs_of_neurons
        std_reciprocities[number - 1] = np.sqrt(
            np.sum(reciprocity_prob_matrix * (1 - reciprocity_prob_matrix)) / num_pairs_of_neurons ** 2)

    fig = plt.figure(figsize=RECT_MEDIUM_FIG_SIZE)
    fontsize = FONT_SIZE
    line_width = LINE_WIDTH
    axes_labelpad = 1
    main_axes = [0.15, 0.16, 0.8, 0.8]
    ax1 = fig.add_axes(main_axes)
    num_stds = 2
    ax1.set_xlim(0, 52)
    ax1.set_ylim(0, 0.02)
    x_axes_ticks = np.arange(0, 55, 10)
    y_axes_ticks = np.arange(0, 0.021, 0.01)
    ax1.set_xticks(x_axes_ticks)
    ax1.set_xticklabels([f'{tick}' for tick in x_axes_ticks], fontsize=fontsize)
    ax1.set_yticks(y_axes_ticks)
    ax1.set_yticklabels([f'{tick:.2f}' for tick in y_axes_ticks], fontsize=fontsize)
    ax1.set_xlabel('number of neuronal types', fontsize=fontsize, labelpad=axes_labelpad)
    ax1.set_ylabel('fraction of reciprocal dyads', fontsize=fontsize, labelpad=axes_labelpad)
    ax1.fill_between(np.arange(1, max_type_number + 1), y1=mean_reciprocities + num_stds * std_reciprocities,
                     y2=mean_reciprocities - num_stds * std_reciprocities, color=SINGLE_EPOCH_INFERRED_COLOR, alpha=0.5,
                     label='model')
    ax1.plot([1, max_type_number], [data_reciprocity, data_reciprocity], lw=line_width, label='data', color='k')
    plt.savefig(os.path.join(out_path, 'panel_a.pdf'), format='pdf')
    plt.show()


def supplement_reciprocity_reciprocal_model(out_path="Figures\\FigS_reciprocity",
                                            saved_calcs_path="SavedOutputs\ReciprocalModel\FullDataset\dyads_distributions\8_types",
                                            is_saved=False, do_save=True):
    num_types = 8
    train_data_path = f"CElegansData\InferredTypes\connectomes\\{num_types}_types\\Dataset7.pkl"
    with open(train_data_path, 'rb') as f:
        data_connectome = pickle.load(f)
    data_reciprocity = calc_reciprocity(data_connectome)
    num_neurons = len(data_connectome.nodes)
    num_synapses = num_neurons * (num_neurons - 1)
    num_pairs_of_neurons = num_synapses / 2
    gamma_range = range(1, 10)
    mean_reciprocities = np.zeros(len(gamma_range))
    std_reciprocities = np.zeros(len(gamma_range))
    with open(f"SavedOutputs\IndependentModel\\average_adj_mats\InferredTypes\\{num_types}_types.pkl", 'rb') as f:
        model_av_adj_mat = pickle.load(f)

    reciprocity_prob_matrix = np.triu(model_av_adj_mat * model_av_adj_mat.T)
    mean_reciprocities[0] = reciprocity_prob_matrix.sum() / num_pairs_of_neurons
    std_reciprocities[0] = np.sqrt(
        np.sum(reciprocity_prob_matrix * (1 - reciprocity_prob_matrix)) / num_pairs_of_neurons ** 2)
    idx = 1
    for gamma in gamma_range[1:]:
        likelihoods_path = f'SavedOutputs\ReciprocalModel\FullDataset\likelihoods\\{num_types}_types\gamma{gamma}'
        smi, beta, _ = find_max_likelihood_full_model(likelihoods_path)
        if not is_saved:
            spls_path = f"SavedOutputs\ReciprocalModel\FullDataset\S+s\\{num_types}_types\gamma{gamma}\\spls_smi{smi:.5f}_beta{beta:.5f}.pkl"
            with open(spls_path, 'rb') as f:
                spls = pickle.load(f)
            model_dyads_distribution = calc_reciprocal_dependence_model_dyads_states_distribution(spls, smi, beta,
                                                                                                  gamma,
                                                                                                  ADULT_WORM_AGE,
                                                                                                  SINGLE_DEVELOPMENTAL_AGE,
                                                                                                  train_data_path)
        else:
            with open(os.path.join(saved_calcs_path, f'gamma{gamma}.pkl'), 'rb') as f:
                model_dyads_distribution = pickle.load(f)
        if do_save:
            with open(os.path.join(saved_calcs_path, f'gamma{gamma}.pkl'), 'wb') as f:
                pickle.dump(model_dyads_distribution, f)

        mean_reciprocities[idx], std_reciprocities[idx] = calc_reciprocal_dependence_model_reciprocity_from_dyads_dist(
            model_dyads_distribution,
            num_neurons)
        idx += 1

    fig = plt.figure(figsize=RECT_MEDIUM_FIG_SIZE)
    fontsize = FONT_SIZE
    line_width = LINE_WIDTH
    axes_labelpad = 1
    main_axes = [0.15, 0.16, 0.8, 0.8]
    ax1 = fig.add_axes(main_axes)
    num_stds = 2
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 0.021)
    x_axes_ticks = np.arange(1, 10, 2)
    y_axes_ticks = np.arange(0, 0.021, 0.01)
    ax1.set_xticks(x_axes_ticks)
    ax1.set_xticklabels([f'{tick}' for tick in x_axes_ticks], fontsize=fontsize)
    ax1.set_yticks(y_axes_ticks)
    ax1.set_yticklabels([f'{tick:.2f}' for tick in y_axes_ticks], fontsize=fontsize)
    ax1.set_xlabel('$\gamma$', fontsize=fontsize, labelpad=axes_labelpad)
    ax1.set_ylabel('fraction of reciprocal dyads', fontsize=fontsize, labelpad=axes_labelpad)
    ax1.fill_between(gamma_range, y1=mean_reciprocities + num_stds * std_reciprocities,
                     y2=mean_reciprocities - num_stds * std_reciprocities, color=SINGLE_EPOCH_INFERRED_COLOR, alpha=0.5,
                     label='model')
    ax1.plot([1, len(gamma_range)], [data_reciprocity, data_reciprocity], lw=line_width, label='data', color='k')
    plt.savefig(os.path.join(out_path, 'panel_b.pdf'), format='pdf')
    plt.show()


def supplement_spl_mats_across_dev(out_path="Figures\\FigS_S+s_across_dev"):
    num_types = 8
    sorted_inferred_types_indices = np.array(
        [INFERRED_TYPES_LABEL_TO_INDEX[label] for label in sorted(INFERRED_TYPES_LABEL_TO_INDEX.keys())]).reshape(
        num_types, 1)

    with open("SavedOutputs\ReciprocalModel\\DyadsSplit\\max_likelihood_params_per_split_3_epochs.pkl",
              'rb') as f:
        max_like_params_multiple = pickle.load(f)

    num_splits = 20
    average_spl_mats_per_stage = []
    for i in range(len(THREE_DEVELOPMENTAL_AGES.keys())):
        average_spl_mats_per_stage.append(np.zeros((num_types, num_types)))

    max_value = np.log10(1)
    min_value = -5
    fontsize = FONT_SIZE
    main_axes = [0.16, 0.18, 0.65, 0.65]
    colorbar_axes = [0.82, 0.18, 0.025, 0.65]
    pylab.rcParams['xtick.major.pad'] = '0.5'
    pylab.rcParams['ytick.major.pad'] = '0.5'
    axis_labelpad = 1
    colorbar_labelpad = 3
    majorticks = np.logspace(min_value, max_value, num=6)
    minorticks = []
    for i in range(1, 6):
        minorticks += list(np.arange(2, 11) / 10 ** i)

    for split in range(1, num_splits + 1):
        smi = max_like_params_multiple[f'split{split}']['S-']
        likelihoods_path = f"SavedOutputs\ReciprocalModel\\DyadsSplit\\likelihoods\ThreeDevStages\split{split}"
        _, beta, _ = find_max_likelihood_full_model(likelihoods_path, smi_value=float(f'{smi:.5f}'))
        spls_file_name = f"spls_smi{smi:.5f}_beta{beta:.5f}.pkl"
        spls_path = os.path.join("SavedOutputs\ReciprocalModel\\DyadsSplit\S+s\ThreeDevStages",
                                 f"split{split}",
                                 spls_file_name)

        for dev_stage in range(len(THREE_DEVELOPMENTAL_AGES.keys())):
            cur_spls_mat, _ = convert_spls_dict_to_mat(spls_path, developmental_stage=dev_stage)
            cur_spls_mat = cur_spls_mat[sorted_inferred_types_indices, sorted_inferred_types_indices.T]
            average_spl_mats_per_stage[dev_stage] += cur_spls_mat

            fig = plt.figure(figsize=SQUARE_FIG_SIZE)
            ax = fig.add_axes(main_axes)
            im = ax.imshow(cur_spls_mat, cmap="Greens", norm=colors.LogNorm(vmin=10 ** min_value, vmax=10 ** max_value))
            cbar_ax = fig.add_axes(colorbar_axes)
            cbar = fig.colorbar(im, cax=cbar_ax, ticks=majorticks)
            cbar.ax.yaxis.set_ticks(minorticks, minor=True)
            cbar_ax.set_yticklabels([r"$10^{{{0:d}}}$".format(i - len(majorticks) + 1) for i in range(len(majorticks))],
                                    fontsize=fontsize)
            cbar.ax.set_title('$S^{+}$', fontsize=fontsize, pad=colorbar_labelpad, loc='left')
            ax.set_ylabel('pre-synaptic cell type', fontsize=fontsize, labelpad=axis_labelpad)
            ax.set_xlabel('post-synaptic cell type', fontsize=fontsize, labelpad=axis_labelpad)
            ax.set_xticks(range(num_types))
            ax.set_xticklabels([f'C{idx}' for idx in range(num_types)], fontsize=fontsize)
            ax.set_yticks(range(num_types))
            ax.set_yticklabels([f'C{idx}' for idx in range(num_types)], fontsize=fontsize)
            plt.savefig(os.path.join(out_path, f'split{split}_stage{dev_stage}.pdf'), format='pdf')
            plt.show(block=False)
            plt.pause(3)
            plt.close()
    for i in range(len(THREE_DEVELOPMENTAL_AGES.keys())):
        average_spl_mat = average_spl_mats_per_stage[i] / num_splits
        fig = plt.figure(figsize=SQUARE_FIG_SIZE)
        ax = fig.add_axes(main_axes)
        im = ax.imshow(average_spl_mat, cmap="Greens", norm=colors.LogNorm(vmin=10 ** min_value, vmax=10 ** max_value))
        cbar_ax = fig.add_axes(colorbar_axes)
        cbar = fig.colorbar(im, cax=cbar_ax, ticks=majorticks)
        cbar.ax.yaxis.set_ticks(minorticks, minor=True)
        cbar_ax.set_yticklabels([r"$10^{{{0:d}}}$".format(i - len(majorticks) + 1) for i in range(len(majorticks))],
                                fontsize=fontsize)
        cbar.ax.set_title('$S^{+}$', fontsize=fontsize, pad=colorbar_labelpad, loc='left')
        ax.set_ylabel('pre-synaptic cell type', fontsize=fontsize, labelpad=axis_labelpad)
        ax.set_xlabel('post-synaptic cell type', fontsize=fontsize, labelpad=axis_labelpad)
        ax.set_xticks(range(num_types))
        ax.set_xticklabels([f'C{idx}' for idx in range(num_types)], fontsize=fontsize)
        ax.set_yticks(range(num_types))
        ax.set_yticklabels([f'C{idx}' for idx in range(num_types)], fontsize=fontsize)
        plt.savefig(os.path.join(out_path, f'average_stage{i}.pdf'), format='pdf')
        plt.show(block=False)
        plt.pause(3)
        plt.close()


if __name__ == "__main__":
    fig_1_b()
    fig_1_c()
    fig_1_d_e_f()
    fig_1_g()

    fig_2_b()
    fig_2_c()
    fig_2_d()
    fig_2_e_f_g()

    fig_graph_features_a()
    fig_graph_features_b()
    fig_graph_features_c_d()
    fig_graph_features_e()
    fig_graph_features_f()

    fig_inferred_types_bio_interpretation_a()
    fig_inferred_types_bio_interpretation_b()
    fig_inferred_types_bio_interpretation_c_d()

    fig_compressed_models_a_b()
    fig_compressed_models_c()
    fig_compressed_models_d()

    fig_6_a()
    fig_6_b()
    fig_6_d()
    fig_6_e()
    fig_6_f()

    worms_overlap()
    prob_hist_by_num_datasets()
    cum_prob_hist_by_num_datasets()
    mean_weighted_connectivity_matrix()
    prediction_vs_num_synapses_data()

    supplement_noisy_birth_times()

    supplement_auc_vs_number_of_inferred_types()
    supplement_outputs_control_overfit()

    supplement_reciprocity_independent_model()
    supplement_reciprocity_reciprocal_model()

    supplement_spl_mats_across_dev()

    supplement_model_variance()
    supplement_model_probs_hist()
