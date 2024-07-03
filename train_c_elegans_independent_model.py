import os
import pickle
import numpy as np
import networkx as nx
from sklearn.metrics import roc_auc_score

from CElegansNeuronsAdder import CElegansNeuronsAdder
from c_elegans_constants import ADULT_WORM_AGE, SINGLE_DEVELOPMENTAL_AGE
from c_elegans_independent_model_training import model_log_likelihood, calc_synaptic_type_factor, \
    average_matrix_log_likelihood, calc_model_adj_mat, convert_spls_dict_to_mat
from compact_model_representaion import find_compact_representation_within_performance_tolerance, \
    find_compression_to_k_values
from c_elegans_data_parsing import SYNAPTIC_COARSE_TYPES
from distance_model import train_types_and_distances_model
from wrap_cluster_runs import find_max_likelihood_full_model
from SynapsesAdder import SynapsesAdder
from SynapsesRemover import SynapsesRemover
from ConnectomeDeveloper import ConnectomeDeveloper

# Indices of the various parameters in each row of the parameters file for calculating likelihoods of the
# model regarding c elegans data.
LIKELIHOOD_SMI_IDX = 0
LIKELIHOOD_BETA_IDX = 1

# Data sets
WORM_ATLAS = "WormAtlas"
MEI_ZHEN_7 = "MeiZhen7"
MEI_ZHEN_8 = "MeiZhen8"


def explore_model_likelihood(param_file_name, developmental_ages, out_dir_path, spls_dir_path, connectome_name,
                             reference_age=ADULT_WORM_AGE, batch_size=1):
    func_id = int(os.environ['LSB_JOBINDEX']) - 1
    func_id = func_id % 1600
    cur_path = os.getcwd()
    out_path = os.path.join(out_dir_path, f"{func_id}.csv")
    with open(os.path.join(cur_path, "ParamFiles", param_file_name), 'rb') as f:
        # Parameters file format: each line is a set of parameters for a different job.
        # columns: S-,beta
        params = pickle.load(f)
    if func_id >= params.shape[0] * batch_size:
        return
    with open(out_path, 'w') as f:
        f.write("smi,beta,log-likelihood\n")
    for i in range(func_id * batch_size, (func_id + 1) * batch_size):
        smi = params[i, LIKELIHOOD_SMI_IDX]
        beta = params[i, LIKELIHOOD_BETA_IDX]
        spls_path = os.path.join(spls_dir_path, f"spls_smi{smi:.5f}_beta{beta:.5f}.pkl")
        with open(spls_path, 'rb') as f:
            spls = pickle.load(f)
        log_likelihood = model_log_likelihood(spls, smi, beta, reference_age, developmental_ages,
                                              os.path.join(cur_path, "CElegansData", connectome_name))

        with open(out_path, 'a') as f:
            f.write(f"{smi},{beta},{log_likelihood}\n")


def calc_spl_per_type_single_developmental_stage(params_path, out_dir_path, types_configuration,
                                                 connectome_file_name, num_artificial_types=8):
    func_id = int(os.environ['LSB_JOBINDEX']) - 1
    func_id = func_id % 1600
    cur_path = os.getcwd()
    with open(params_path, 'rb') as f:
        params = pickle.load(f)
    smi = params[func_id, 0]
    beta = params[func_id, 1]
    c_elegans_data_path = os.path.join(cur_path, "CElegansData")

    if CElegansNeuronsAdder.SINGLE_TYPE == types_configuration:
        synaptic_types_dict = {(None, None): 1}
    elif CElegansNeuronsAdder.COARSE_TYPES == types_configuration:
        synaptic_types_dict = SYNAPTIC_COARSE_TYPES.copy()
    elif CElegansNeuronsAdder.SUB_TYPES == types_configuration:
        synaptic_types_dict_path = os.path.join(cur_path, "CElegansData", "SubTypes", "synaptic_subtypes_dict.pkl")
        with open(synaptic_types_dict_path, 'rb') as f:
            synaptic_types_dict = pickle.load(f)
    elif CElegansNeuronsAdder.ARTIFICIAL_TYPES == types_configuration:
        counter = 0
        synaptic_types_dict = {}
        for i in range(num_artificial_types):
            for j in range(num_artificial_types):
                synaptic_types_dict[(i, j)] = counter
                counter += 1
    spls = {0: synaptic_types_dict.copy()}
    for syn_type in synaptic_types_dict.keys():
        spls[0][syn_type], _ = calc_synaptic_type_factor(syn_type, beta, smi, ADULT_WORM_AGE, SINGLE_DEVELOPMENTAL_AGE,
                                                         spls.copy(),
                                                         types_configuration,
                                                         os.path.join(c_elegans_data_path, connectome_file_name))

    with open(os.path.join(out_dir_path, f"spls_smi{smi:.5f}_beta{beta:.5f}.pkl"), 'wb') as f:
        pickle.dump(spls, f)


def train_types_and_distances_model_distributed(type_configuration, connectome_name, out_dir_path, neurons_subset_path,
                                                data_dir_path=os.path.join(os.getcwd(), "CElegansData"),
                                                num_artificial_types=1,
                                                types_file_name=CElegansNeuronsAdder.DEFAULT_TYPES_FILE_NAME):
    func_id = int(os.environ['LSB_JOBINDEX']) - 1
    if func_id > 160:
        return
    examined_value = func_id / 40
    average_mat = train_types_and_distances_model(examined_value, type_configuration, connectome_name,
                                                  neurons_subset_path,
                                                  data_dir_path=data_dir_path,
                                                  num_artificial_types=num_artificial_types,
                                                  types_file_name=types_file_name)
    if CElegansNeuronsAdder.SINGLE_TYPE == type_configuration:
        likelihoods_path = os.path.join(out_dir_path, "likelihoods", "SingleType")
        average_conectomes_path = os.path.join(out_dir_path, "average_adj_mats", "SingleType")
    elif CElegansNeuronsAdder.COARSE_TYPES == type_configuration:
        likelihoods_path = os.path.join(out_dir_path, "likelihoods", "CoarseTypes")
        average_conectomes_path = os.path.join(out_dir_path, "average_adj_mats", "CoarseTypes")
    elif CElegansNeuronsAdder.SUB_TYPES == type_configuration:
        likelihoods_path = os.path.join(out_dir_path, "likelihoods", "SubTypes")
        average_conectomes_path = os.path.join(out_dir_path, "average_adj_mats", "SubTypes")
    try:
        os.mkdir(likelihoods_path)
    except FileExistsError:
        pass
    try:
        os.mkdir(average_conectomes_path)
    except FileExistsError:
        pass
    beta_log_like_path = os.path.join(likelihoods_path, f"{examined_value:.3f}_beta_log_like.csv")
    average_adj_mat_path = os.path.join(average_conectomes_path, f"{examined_value:.3f}_average_adj_mat.pkl")
    with open(beta_log_like_path, 'w') as f:
        f.write("beta,log-likelihood\n")
    with open(average_adj_mat_path, 'wb') as f:
        pickle.dump(average_mat, f)
    log_likelihood = average_matrix_log_likelihood(average_mat, os.path.join(data_dir_path, connectome_name))
    with open(beta_log_like_path, 'a') as f:
        f.write(f"{examined_value},{log_likelihood}\n")


def calc_average_adj_mat_inferred_types_distributed(
        out_path=os.path.join("SavedOutputs", "IndependentModel", "average_adj_mats", "InferredTypes")):
    num_types = int(os.environ['LSB_JOBINDEX'])
    cur_path = os.getcwd()
    connectome_name = os.path.join(cur_path, "CElegansData", "InferredTypes", "FullDataset", "connectomes",
                                   f"{num_types}_types", "Dataset7.pkl")
    spls_path = os.path.join(cur_path, "SavedOutputs", "IndependentModel", "S+s", "InferredTypes", f"{num_types}_types")
    likelihood_path = os.path.join(cur_path, "SavedOutputs", "IndependentModel", "likelihoods", "InferredTypes",
                                   f"{num_types}_types")

    smi, beta, _ = find_max_likelihood_full_model(likelihood_path)
    spls_path = os.path.join(spls_path, f"spls_smi{smi:.5f}_beta{beta:.5f}.pkl")
    with open(spls_path, 'rb') as f:
        spls = pickle.load(f)

    average_mat = calc_model_adj_mat(spls, smi, beta, ADULT_WORM_AGE, SINGLE_DEVELOPMENTAL_AGE, connectome_name)

    with open(os.path.join(out_path, f"{num_types}_types.pkl"), 'wb') as f:
        pickle.dump(average_mat, f)


def train_on_control_outputs():
    cur_path = os.getcwd()
    func_id = int(os.environ['LSB_JOBINDEX']) - 1
    num_learned_types = func_id // 1600 + 1
    num_control_type = 8

    connectome_name = os.path.join("InferredTypes", "ModelSamplesControls", f"{num_control_type}_types",
                                   f"{num_learned_types}_types.pkl")

    control_types_dir_path = os.path.join(cur_path, "SavedOutputs", "ModelOutputsControl", "S+s",
                                          f"{num_control_type}_control_types")
    if not os.path.exists(control_types_dir_path):
        try:
            os.mkdir(control_types_dir_path)
        except FileExistsError:
            pass
    spls_path = os.path.join(control_types_dir_path, f'{num_learned_types}_learned_types')
    if not os.path.exists(spls_path):
        try:
            os.mkdir(spls_path)
        except FileExistsError:
            pass
    calc_spl_per_type_single_developmental_stage(
        os.path.join(cur_path, "ParamFiles", "spl_per_type_params_low_smi_low_beta.pkl"),
        spls_path,
        CElegansNeuronsAdder.ARTIFICIAL_TYPES, connectome_file_name=connectome_name,
        num_artificial_types=num_learned_types)

    control_types_dir_path = os.path.join(cur_path, "SavedOutputs", "ModelOutputsControl", "likelihoods",
                                          f"{num_control_type}_control_types")
    if not os.path.exists(control_types_dir_path):
        try:
            os.mkdir(control_types_dir_path)
        except FileExistsError:
            pass
    likelihood_single_path = os.path.join(control_types_dir_path, f'{num_learned_types}_learned_types')
    if not os.path.exists(likelihood_single_path):
        try:
            os.mkdir(likelihood_single_path)
        except FileExistsError:
            pass
    explore_model_likelihood("spl_per_type_params_low_smi_low_beta.pkl",
                             SINGLE_DEVELOPMENTAL_AGE,
                             likelihood_single_path,
                             spls_path, connectome_name, batch_size=1)


def train_single_stage_sub_types_mei_zhen(data_set, params_file_name="spl_per_type_params_low_smi_low_beta.pkl"):
    cur_path = os.getcwd()
    connectome_name = os.path.join("SubTypes", "FullDataset", "connectomes", f"Dataset{data_set}.pkl")

    spls_single_path = os.path.join(cur_path, "SavedOutputs", "IndependentModel", "S+s", "SubTypes")
    calc_spl_per_type_single_developmental_stage(
        os.path.join(cur_path, "ParamFiles", params_file_name), spls_single_path,
        CElegansNeuronsAdder.SUB_TYPES, connectome_file_name=connectome_name)
    likelihood_single_path = os.path.join(cur_path, "SavedOutputs", "IndependentModel", "likelihoods", "SubTypes")
    explore_model_likelihood(params_file_name, SINGLE_DEVELOPMENTAL_AGE, likelihood_single_path, spls_single_path,
                             connectome_name, batch_size=1)


def train_single_stage_artificial_types_mei_zhen_7(params_file_name="spl_per_type_params_low_smi_low_beta.pkl"):
    cur_path = os.getcwd()
    func_id = int(os.environ['LSB_JOBINDEX']) - 1
    num_artificial_types = func_id // 1600 + 1
    connectome_name = os.path.join("InferredTypes", "FullDataset", "connectomes",
                                   f"{num_artificial_types}_types", "Dataset7.pkl")

    spls_single_path = os.path.join(cur_path, "SavedOutputs", "IndependentModel", "S+s", "InferredTypes",
                                    f"{num_artificial_types}_types")
    if not os.path.exists(spls_single_path):
        try:
            os.mkdir(spls_single_path)
        except FileExistsError:
            pass
    calc_spl_per_type_single_developmental_stage(
        os.path.join(cur_path, "ParamFiles", params_file_name),
        spls_single_path,
        CElegansNeuronsAdder.ARTIFICIAL_TYPES, connectome_file_name=connectome_name,
        num_artificial_types=num_artificial_types)
    likelihood_single_path = os.path.join(cur_path, "SavedOutputs", "IndependentModel", "likelihoods", "InferredTypes",
                                          f"{num_artificial_types}_types")
    if not os.path.exists(likelihood_single_path):
        try:
            os.mkdir(likelihood_single_path)
        except FileExistsError:
            pass
    explore_model_likelihood(params_file_name, SINGLE_DEVELOPMENTAL_AGE,
                             likelihood_single_path,
                             spls_single_path, connectome_name, batch_size=1)


def train_single_stage_random_types_mei_zhen():
    cur_path = os.getcwd()
    func_id = int(os.environ['LSB_JOBINDEX']) - 1
    num_types = func_id // 1600 + 1
    connectome_name = os.path.join("RandomTypes", "FullDataset", "connectomes", f"{num_types}_types",
                                   "Dataset7.pkl")

    spls_single_path = os.path.join(cur_path, "SavedOutputs", "IndependentModel", "S+s", "RandomTypes",
                                    f"{num_types}_types")
    if not os.path.exists(spls_single_path):
        try:
            os.mkdir(spls_single_path)
        except FileExistsError:
            pass
    params_file_name = "spl_per_type_params_low_smi_low_beta.pkl"
    calc_spl_per_type_single_developmental_stage(
        os.path.join(cur_path, "ParamFiles", params_file_name),
        spls_single_path,
        CElegansNeuronsAdder.ARTIFICIAL_TYPES, connectome_file_name=connectome_name,
        num_artificial_types=num_types)
    likelihood_single_path = os.path.join(cur_path, "SavedOutputs", "IndependentModel", "likelihoods", "RandomTypes",
                                          f"{num_types}_types")
    if not os.path.exists(likelihood_single_path):
        try:
            os.mkdir(likelihood_single_path)
        except FileExistsError:
            pass
    explore_model_likelihood(params_file_name, SINGLE_DEVELOPMENTAL_AGE,
                             likelihood_single_path,
                             spls_single_path, connectome_name, batch_size=1)


def train_sub_models_sub_types_mei_zhen(data_set):
    cur_path = os.getcwd()

    distances_and_type_out_path = os.path.join(cur_path, "SavedOutputs", "DistancesModel")
    neuron_subset_path = os.path.join(cur_path, "CElegansData", "nerve_ring_neurons_subset.pkl")

    train_single_stage_sub_types_mei_zhen(data_set)

    connectome_name = os.path.join("SingleType", "FullDataset", "connectomes", f"Dataset{data_set}.pkl")
    type_configuration = CElegansNeuronsAdder.SINGLE_TYPE
    train_types_and_distances_model_distributed(type_configuration, connectome_name, distances_and_type_out_path,
                                                neuron_subset_path)
    spls_single_path = os.path.join(cur_path, "SavedOutputs", "IndependentModel", "S+s", "SingleType")
    calc_spl_per_type_single_developmental_stage(
        os.path.join(cur_path, "ParamFiles", "spl_per_type_params_low_smi_low_beta.pkl"),
        spls_single_path,
        CElegansNeuronsAdder.SINGLE_TYPE, connectome_file_name=connectome_name)
    likelihood_single_path = os.path.join(cur_path, "SavedOutputs", "IndependentModel", "likelihoods", "SingleType")
    explore_model_likelihood("spl_per_type_params_low_smi_low_beta.pkl", SINGLE_DEVELOPMENTAL_AGE,
                             likelihood_single_path,
                             spls_single_path, connectome_name, batch_size=1)

    connectome_name = os.path.join("CoarseTypes", "FullDataset", "connectomes", f"Dataset{data_set}.pkl")
    spls_coarse_path = os.path.join(cur_path, "SavedOutputs", "IndependentModel", "S+s", "CoarseTypes")
    calc_spl_per_type_single_developmental_stage(
        os.path.join(cur_path, "ParamFiles", "spl_per_type_params_low_smi_low_beta.pkl"),
        spls_coarse_path,
        CElegansNeuronsAdder.COARSE_TYPES, connectome_file_name=connectome_name)
    likelihood_coarse_path = os.path.join(cur_path, "SavedOutputs", "IndependentModel", "likelihoods", "CoarseTypes")
    explore_model_likelihood("spl_per_type_params_low_smi_low_beta.pkl", SINGLE_DEVELOPMENTAL_AGE,
                             likelihood_coarse_path,
                             spls_coarse_path, connectome_name, batch_size=1)


def find_compact_representations():
    cur_path = os.getcwd()
    func_id = int(os.environ['LSB_JOBINDEX'])
    performance_tolerance_fraction = max(func_id // 30 * 0.05, 0.01)
    num_types = (func_id - 1) % 30 + 1
    likelihood_path = os.path.join(cur_path, "SavedOutputs", "IndependentModel", "likelihoods", "InferredTypes",
                                   f"{num_types}_types")
    smi, beta, _ = find_max_likelihood_full_model(likelihood_path)
    train_data_path = os.path.join(cur_path, "CElegansData", "InferredTypes", "FullDataset", "connectomes",
                                   f"{num_types}_types", "Dataset7.pkl")
    test_data_path = os.path.join(cur_path, "CElegansData", "InferredTypes", "FullDataset", "connectomes",
                                  f"{num_types}_types", "Dataset8.pkl")
    spls_dir_path = os.path.join(cur_path, "SavedOutputs", "IndependentModel", "S+s", "InferredTypes",
                                 f"{num_types}_types")
    compact_size, compact_spls, compact_model_av_mat, unex_var_thr = find_compact_representation_within_performance_tolerance(
        smi, beta,
        spls_dir_path,
        train_data_path,
        test_data_path,
        performance_tolerance_fraction)
    out_path = os.path.join(cur_path, "SavedOutputs", "IndependentModel", "compact_models",
                            f"{performance_tolerance_fraction:.2f}_performance_decrease")
    if not os.path.exists(out_path):
        try:
            os.mkdir(out_path)
        except FileExistsError:
            pass
    out_path = os.path.join(out_path, f"{num_types}")
    if not os.path.exists(out_path):
        try:
            os.mkdir(out_path)
        except FileExistsError:
            pass
    with open(os.path.join(out_path, 'compact_size.pkl'), 'wb') as f:
        pickle.dump(compact_size, f)
    with open(os.path.join(out_path, 'compact_spls.pkl'), 'wb') as f:
        pickle.dump(compact_spls, f)
    with open(os.path.join(out_path, 'compact_model_av_mat.pkl'), 'wb') as f:
        pickle.dump(compact_model_av_mat, f)
    with open(os.path.join(out_path, 'unexplained_variance_ratio_threshold.pkl'), 'wb') as f:
        pickle.dump(unex_var_thr, f)


def calc_average_connectome_noisy_birth_times():
    cur_path = os.getcwd()
    func_id = int(os.environ['LSB_JOBINDEX'])
    noise_range = np.arange(0.1, 1.1, 0.1)
    likelihoods_path = os.path.join(cur_path, "SavedOutputs", "IndependentModel", "likelihoods", "SubTypes")
    smi, beta, _ = find_max_likelihood_full_model(likelihoods_path)
    spls_path = os.path.join(cur_path, "SavedOutputs", "IndependentModel", "S+s", "SubTypes",
                             f'spls_smi{smi:.5f}_beta{beta:.5f}.pkl')
    with open(spls_path, 'rb') as f:
        spls = pickle.load(f)
    for noise_level in noise_range:
        noisy_birth_times_data_path = os.path.join(cur_path, "CElegansData", "SubTypes", "FullDataset",
                                                   "noised_birth_times_connectomes", f'{int(100 * noise_level)}%_noise',
                                                   f'{func_id}', "Dataset7.pkl")
        out_path = os.path.join(cur_path, "SavedOutputs", "IndependentModel", "average_adj_mats", "SubTypes",
                                "noised_birth_times", f'{int(100 * noise_level)}%_noise')
        if not os.path.exists(out_path):
            try:
                os.mkdir(out_path)
            except FileExistsError:
                pass

        average_mat = calc_model_adj_mat(spls, smi, beta, ADULT_WORM_AGE, SINGLE_DEVELOPMENTAL_AGE,
                                         noisy_birth_times_data_path)

        with open(os.path.join(out_path, f'{func_id}.pkl'), 'wb') as f:
            pickle.dump(average_mat, f)


def run_c_elegans_model_distributed():
    cur_path = os.getcwd()
    num_types = 8
    likelihoods_path = os.path.join(cur_path, "SavedOutputs", "IndependentModel", "likelihoods", "InferredTypes",
                                    f"{num_types}_types")
    smi, beta, _ = find_max_likelihood_full_model(likelihoods_path)
    type_configuration = CElegansNeuronsAdder.ARTIFICIAL_TYPES
    spls_file_name = f"spls_smi{smi:.5f}_beta{beta:.5f}.pkl"
    spls_path = os.path.join(cur_path, "SavedOutputs", "IndependentModel", "S+s", "InferredTypes", f"{num_types}_types",
                             spls_file_name)
    with open(spls_path, 'rb') as f:
        spls = pickle.load(f)

    func_id = int(os.environ['LSB_JOBINDEX']) - 1
    out_path = os.path.join(cur_path, "SavedOutputs", "IndependentModel", "connectomes", "InferredTypes",
                            f"{num_types}_types")
    if not os.path.exists(out_path):
        try:
            os.mkdir(out_path)
        except FileExistsError:
            pass

    ops_list = [CElegansNeuronsAdder(type_configuration,
                                     c_elegans_data_path=os.path.join(cur_path, "CElegansData"),
                                     types_file_name=os.path.join("InferredTypes", "types", f"{num_types}.pkl"),
                                     neurons_sub_set_path=os.path.join("CElegansData",
                                                                       "nerve_ring_neurons_subset.pkl")),
                SynapsesAdder(spls, SINGLE_DEVELOPMENTAL_AGE, beta, reciprocal_factor=1),
                SynapsesRemover(smi)]
    cd = ConnectomeDeveloper(ops_list, time_step=10, initial_state=nx.empty_graph(0, create_using=nx.DiGraph()),
                             initial_node_attributes={})
    cd.simulate(ADULT_WORM_AGE)
    connectome = cd.get_connectome()

    with open(os.path.join(out_path, f"{func_id}.pkl"), 'wb') as f:
        pickle.dump(connectome, f)


def calc_performance_of_compressed_models():
    cur_path = os.getcwd()
    num_types = 8
    compressed_num_spls = int(os.environ['LSB_JOBINDEX'])
    out_path = os.path.join(cur_path, "SavedOutputs", "IndependentModel", "compact_models",
                            f"{num_types}_types_model_compressions",
                            f"{compressed_num_spls}_spl_params")
    if not os.path.exists(out_path):
        try:
            os.mkdir(out_path)
        except FileExistsError:
            pass
    likelihoods_path = os.path.join(cur_path, "SavedOutputs", "IndependentModel", "likelihoods", "InferredTypes",
                                    f"{num_types}_types")
    smi, beta, _ = find_max_likelihood_full_model(likelihoods_path)
    spls_file_name = f"spls_smi{smi:.5f}_beta{beta:.5f}.pkl"
    spls_path = os.path.join(cur_path, "SavedOutputs", "IndependentModel", "S+s", "InferredTypes", f"{num_types}_types",
                             spls_file_name)
    train_data_path = os.path.join(cur_path, "CElegansData", "InferredTypes", "FullDataset", "connectomes",
                                   f"{num_types}_types", "Dataset7.pkl")
    test_data_path = os.path.join(cur_path, "CElegansData", "InferredTypes", "FullDataset", "connectomes",
                                  f"{num_types}_types", "Dataset8.pkl")

    compact_spls = find_compression_to_k_values(spls_path, compressed_num_spls)
    compact_model_av_mat = calc_model_adj_mat(compact_spls, smi, beta, ADULT_WORM_AGE,
                                              SINGLE_DEVELOPMENTAL_AGE,
                                              train_data_path)
    with open(test_data_path, 'rb') as f:
        test_data = pickle.load(f)
    test_data_mat = nx.to_numpy_array(test_data, nodelist=sorted(test_data.nodes))
    test_data_mat = test_data_mat.astype(int)
    auc = roc_auc_score(test_data_mat.flatten(), compact_model_av_mat.flatten())
    with open(os.path.join(out_path, 'compact_spls.pkl'), 'wb') as f:
        pickle.dump(compact_spls, f)
    with open(os.path.join(out_path, 'compact_model_av_mat.pkl'), 'wb') as f:
        pickle.dump(compact_model_av_mat, f)
    with open(os.path.join(out_path, 'auc.pkl'), 'wb') as f:
        pickle.dump(auc, f)


def main():
    train_on_control_outputs()

    train_sub_models_sub_types_mei_zhen(7)

    train_single_stage_artificial_types_mei_zhen_7()

    train_single_stage_random_types_mei_zhen()

    calc_average_adj_mat_inferred_types_distributed()

    find_compact_representations()

    calc_performance_of_compressed_models()

    calc_average_connectome_noisy_birth_times()

    run_c_elegans_model_distributed()

    return 0


if __name__ == "__main__":
    main()
