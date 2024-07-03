import os
import pickle
import networkx as nx

from c_elegans_reciprocal_model_training import search_spl_reciprocal_dependence_model, \
    search_spl_reciprocal_dependence_model_syn_list, calc_reciprocal_dependence_model_log_likelihood, \
    calc_reciprocal_dependence_model_log_likelihood_syn_list, \
    calc_reciprocal_dependence_model_dyads_states_distribution_syn_list, \
    save_max_like_params_per_split_single_epoch, save_max_like_params_per_split_three_epochs
from c_elegans_constants import ADULT_WORM_AGE, SINGLE_DEVELOPMENTAL_AGE, THREE_DEVELOPMENTAL_AGES_CONSISTENT, \
    FULL_DEVELOPMENT_AGES_CONSISTENT
from CElegansNeuronsAdder import CElegansNeuronsAdder
from SynapsesAdder import SynapsesAdder
from SynapsesRemover import SynapsesRemover
from ConnectomeDeveloper import ConnectomeDeveloper
from wrap_cluster_runs import find_max_likelihood_full_model


def search_spl_per_type_reciprocity_dependence_model_distributed_single_stage_full_dataset(params_path, gamma,
                                                                                           out_dir_path,
                                                                                           num_artificial_types,
                                                                                           connectome_name):
    func_id = int(os.environ['LSB_JOBINDEX']) - 1
    func_id = func_id % 1600
    cur_path = os.getcwd()
    with open(params_path, 'rb') as f:
        params = pickle.load(f)
    smi = params[func_id, 0]
    beta = params[func_id, 1]

    c_elegans_data_path = os.path.join(cur_path, "CElegansData")
    connectome_path = os.path.join(c_elegans_data_path, connectome_name)

    counter = 0
    synaptic_types_dict = {}
    for i in range(num_artificial_types):
        for j in range(num_artificial_types):
            synaptic_types_dict[(i, j)] = counter
            counter += 1

    spls = {0: synaptic_types_dict.copy()}

    calculated_neuronal_type_couples = []
    for syn_type in synaptic_types_dict.keys():
        syn_type_forth = syn_type
        syn_type_back = (syn_type[1], syn_type[0])
        if syn_type_back in calculated_neuronal_type_couples:
            continue
        spls[0][syn_type_forth], spls[0][syn_type_back] = search_spl_reciprocal_dependence_model(syn_type_forth, smi,
                                                                                                 beta,
                                                                                                 gamma,
                                                                                                 ADULT_WORM_AGE,
                                                                                                 SINGLE_DEVELOPMENTAL_AGE,
                                                                                                 spls,
                                                                                                 CElegansNeuronsAdder.ARTIFICIAL_TYPES,
                                                                                                 connectome_path)
        calculated_neuronal_type_couples.append(syn_type_forth)
        calculated_neuronal_type_couples.append(syn_type_back)

    with open(os.path.join(out_dir_path, f"spls_smi{smi:.5f}_beta{beta:.5f}.pkl"), 'wb') as f:
        pickle.dump(spls, f)


def search_spl_per_type_reciprocity_dependence_model_distributed_syn_list(params_path, gamma, out_dir_path,
                                                                          num_artificial_types,
                                                                          dev_stages, full_developmental_ages,
                                                                          all_data_sets_path):
    func_id = int(os.environ['LSB_JOBINDEX']) - 1
    func_id = func_id % 1600
    cur_path = os.getcwd()
    with open(params_path, 'rb') as f:
        params = pickle.load(f)
    smi = params[func_id, 0]
    beta = params[func_id, 1]

    c_elegans_data_path = os.path.join(cur_path, "CElegansData")

    counter = 0
    synaptic_types_dict = {}
    for i in range(num_artificial_types):
        for j in range(num_artificial_types):
            synaptic_types_dict[(i, j)] = counter
            counter += 1

    all_data_sets_path = os.path.join(c_elegans_data_path, all_data_sets_path)
    spls_across_development = {}
    data_set_idx = 1
    data_sets_list = sorted(os.listdir(all_data_sets_path))
    data_sets_list_copy = data_sets_list.copy()
    for data_set in data_sets_list_copy:
        is_relevant = False
        for stage in dev_stages:
            if str(stage + 1) in data_set:
                is_relevant = True
        if not is_relevant:
            data_sets_list.remove(data_set)
    for data_set in data_sets_list:
        data_set_path = os.path.join(all_data_sets_path, data_set)
        with open(data_set_path, 'rb') as f:
            synapses_list = pickle.load(f)
        cur_ages = {}
        age_idx = 0
        for prev_data_set_idx in range(data_set_idx):
            prev_data_set = data_sets_list[prev_data_set_idx]
            stage = int(prev_data_set[
                        prev_data_set.find("Dataset") + len("Dataset"): prev_data_set.find("Dataset") + len(
                            "Dataset") + 1])
            stage = min(stage, 7)
            cur_ages[age_idx] = full_developmental_ages[stage - 1]
            age_idx += 1
        spls_across_development[data_set_idx - 1] = synaptic_types_dict.copy()
        calculated_neuronal_type_couples = []
        for syn_type in synaptic_types_dict.keys():
            syn_type_forth = syn_type
            syn_type_back = (syn_type[1], syn_type[0])
            if syn_type_back in calculated_neuronal_type_couples or syn_type_forth in calculated_neuronal_type_couples:
                continue
            spls_across_development[data_set_idx - 1][syn_type_forth], spls_across_development[data_set_idx - 1][
                syn_type_back] = search_spl_reciprocal_dependence_model_syn_list(syn_type_forth, smi, beta, gamma,
                                                                                 cur_ages[age_idx - 1],
                                                                                 cur_ages, spls_across_development,
                                                                                 synapses_list)
            calculated_neuronal_type_couples.append(syn_type_forth)
            calculated_neuronal_type_couples.append(syn_type_back)
        data_set_idx += 1

    with open(os.path.join(out_dir_path, f"spls_smi{smi:.5f}_beta{beta:.5f}.pkl"), 'wb') as f:
        pickle.dump(spls_across_development, f)


def calc_reciprocal_dependence_model_likelihood_distributed_single_stage_full_dataset(param_file_name, gamma,
                                                                                      out_dir_path,
                                                                                      spls_dir_path,
                                                                                      connectome_name,
                                                                                      batch_size=1):
    func_id = int(os.environ['LSB_JOBINDEX']) - 1
    func_id = func_id % 1600
    cur_path = os.getcwd()
    c_elegans_data_path = os.path.join(cur_path, "CElegansData")
    out_path = os.path.join(out_dir_path, f"{func_id}.csv")
    with open(param_file_name, 'rb') as f:
        # Parameters file format: each line is a set of parameters for a different job.
        # columns: S-,beta
        params = pickle.load(f)
    if func_id >= params.shape[0] * batch_size:
        return
    with open(out_path, 'w') as f:
        f.write("smi,beta,gamma,log-likelihood\n")
    for i in range(func_id * batch_size, (func_id + 1) * batch_size):
        smi = params[i, 0]
        beta = params[i, 1]
        spls_path = os.path.join(spls_dir_path, f"spls_smi{smi:.5f}_beta{beta:.5f}.pkl")
        with open(spls_path, 'rb') as f:
            spls = pickle.load(f)
        log_likelihood = calc_reciprocal_dependence_model_log_likelihood(spls, smi, beta, gamma, ADULT_WORM_AGE,
                                                                         SINGLE_DEVELOPMENTAL_AGE,
                                                                         os.path.join(c_elegans_data_path,
                                                                                      connectome_name))
        with open(out_path, 'a') as f:
            f.write(f"{smi},{beta},{gamma},{log_likelihood}\n")


def calc_reciprocal_dependence_model_likelihood_distributed_syn_list(param_file_name, gamma, developmental_ages,
                                                                     out_dir_path, spls_dir_path, syn_list_path):
    func_id = int(os.environ['LSB_JOBINDEX']) - 1
    func_id = func_id % 1600
    cur_path = os.getcwd()
    c_elegans_data_path = os.path.join(cur_path, "CElegansData")
    out_path = os.path.join(out_dir_path, f"{func_id}.csv")
    with open(param_file_name, 'rb') as f:
        # Parameters file format: each line is a set of parameters for a different job.
        # columns: S-,beta
        params = pickle.load(f)
    if func_id >= params.shape[0]:
        return
    with open(out_path, 'w') as f:
        f.write("smi,beta,gamma,log-likelihood\n")
    for i in range(func_id, (func_id + 1)):
        smi = params[i, 0]
        beta = params[i, 1]
        spls_path = os.path.join(spls_dir_path, f"spls_smi{smi:.5f}_beta{beta:.5f}.pkl")
        with open(spls_path, 'rb') as f:
            spls = pickle.load(f)
        log_likelihood = calc_reciprocal_dependence_model_log_likelihood_syn_list(spls, smi, beta, gamma,
                                                                                  ADULT_WORM_AGE,
                                                                                  developmental_ages,
                                                                                  os.path.join(c_elegans_data_path,
                                                                                               syn_list_path))
        with open(out_path, 'a') as f:
            f.write(f"{smi},{beta},{gamma},{log_likelihood}\n")


def train_reciprocity_dependence_model_distributed_single_stage_full_dataset(num_artificial_types=8):
    cur_path = os.getcwd()
    params_path = os.path.join(cur_path, "ParamFiles", "spl_per_type_params_low_smi_low_beta.pkl")
    all_data_sets_path = os.path.join(cur_path, "CElegansData", "InferredTypes", "FullDataset", "connectomes",
                                      f"{num_artificial_types}_types")
    func_id = int(os.environ['LSB_JOBINDEX']) - 1
    gamma = (func_id // 1600 + 2)
    out_dir_path_spls = os.path.join(cur_path, "SavedOutputs", "ReciprocalModel", "FullDataset", "S+s",
                                     f"{num_artificial_types}_types", f"gamma{gamma}")
    if not os.path.exists(out_dir_path_spls):
        try:
            os.mkdir(out_dir_path_spls)
        except FileExistsError:
            pass
    search_spl_per_type_reciprocity_dependence_model_distributed_single_stage_full_dataset(params_path, gamma,
                                                                                           out_dir_path_spls,
                                                                                           num_artificial_types,
                                                                                           os.path.join(
                                                                                               all_data_sets_path,
                                                                                               "Dataset7.pkl"))
    out_dir_path_likelihoods = os.path.join(cur_path, "SavedOutputs", "ReciprocalModel", "FullDataset", "likelihoods",
                                            f"{num_artificial_types}_types", f"gamma{gamma}")
    if not os.path.exists(out_dir_path_likelihoods):
        try:
            os.mkdir(out_dir_path_likelihoods)
        except FileExistsError:
            pass
    calc_reciprocal_dependence_model_likelihood_distributed_single_stage_full_dataset(params_path, gamma,
                                                                                      out_dir_path_likelihoods,
                                                                                      out_dir_path_spls,
                                                                                      os.path.join(all_data_sets_path,
                                                                                                   "Dataset7.pkl"))


def train_reciprocity_dependence_model_distributed_syn_list(num_artificial_types, dev_stages, all_data_sets_path,
                                                            dev_stages_str, gamma=7):
    cur_path = os.getcwd()
    params_path = os.path.join(cur_path, "ParamFiles", "spl_per_type_params_low_smi_low_beta.pkl")
    split = (int(os.environ['LSB_JOBINDEX']) - 1) // 1600 + 1
    out_dir_path_spls = os.path.join(cur_path, "SavedOutputs", "ReciprocalModel", "BirthTimesConsistent", "S+s",
                                     dev_stages_str, f"split{split}")
    all_data_sets_path = os.path.join(all_data_sets_path, f'split{split}', 'train')
    if not os.path.exists(out_dir_path_spls):
        try:
            os.mkdir(out_dir_path_spls)
        except FileExistsError:
            pass
    search_spl_per_type_reciprocity_dependence_model_distributed_syn_list(params_path, gamma, out_dir_path_spls,
                                                                          num_artificial_types, dev_stages,
                                                                          FULL_DEVELOPMENT_AGES_CONSISTENT,
                                                                          all_data_sets_path)
    out_dir_path_likelihoods = os.path.join(cur_path, "SavedOutputs", "ReciprocalModel", "BirthTimesConsistent",
                                            "likelihoods", dev_stages_str, f"split{split}")
    if not os.path.exists(out_dir_path_likelihoods):
        try:
            os.mkdir(out_dir_path_likelihoods)
        except FileExistsError:
            pass
    dev_ages = {}
    idx = 0
    for stage in dev_stages:
        cur_stage = min(stage, 6)
        dev_ages[idx] = FULL_DEVELOPMENT_AGES_CONSISTENT[cur_stage]
        idx += 1
    calc_reciprocal_dependence_model_likelihood_distributed_syn_list(params_path, gamma, dev_ages,
                                                                     out_dir_path_likelihoods,
                                                                     out_dir_path_spls,
                                                                     os.path.join(all_data_sets_path, "Dataset8.pkl"))


def run_reciprocity_dependence_model_distributed_full_dataset_single_dev_stage(spls_path, smi, beta, gamma,
                                                                               out_dir_path, num_artificial_types):
    func_id = int(os.environ['LSB_JOBINDEX'])
    cur_path = os.getcwd()
    time_step = 10
    with open(spls_path, 'rb') as f:
        spls = pickle.load(f)
    c_elegans_data_path = os.path.join(cur_path, "CElegansData")
    types_path = os.path.join(c_elegans_data_path, "InferredTypes", "types", f"{num_artificial_types}.pkl")
    neurons_subset_path = os.path.join(c_elegans_data_path, "nerve_ring_neurons_subset.pkl")
    syn_adder = SynapsesAdder(spls, SINGLE_DEVELOPMENTAL_AGE, beta, gamma)
    ops_list = [CElegansNeuronsAdder(CElegansNeuronsAdder.ARTIFICIAL_TYPES, c_elegans_data_path=c_elegans_data_path,
                                     types_file_name=types_path, neurons_sub_set_path=neurons_subset_path),
                syn_adder,
                SynapsesRemover(smi)]
    cd = ConnectomeDeveloper(ops_list, time_step=time_step, initial_state=nx.empty_graph(0, create_using=nx.DiGraph()),
                             initial_node_attributes={})
    cd.simulate(ADULT_WORM_AGE)
    connectome = cd.get_connectome()
    with open(os.path.join(out_dir_path, f"{func_id}.pkl"), 'wb') as f:
        pickle.dump(connectome, f)


def run_reciprocity_dependence_model_distributed_across_gammas_full_dataset_single_dev_stage(gamma_range):
    cur_path = os.getcwd()

    num_artificial_types = 8

    for gamma in gamma_range:
        likelihoods_path = os.path.join(cur_path, "SavedOutputs", "ReciprocalModel", "FullDataset", "likelihoods",
                                        f"{num_artificial_types}_types", f"gamma{gamma}")
        smi, beta, _ = find_max_likelihood_full_model(likelihoods_path)
        spls_path = os.path.join(cur_path, "SavedOutputs", "ReciprocalModel", "FullDataset", "S+s",
                                 f"{num_artificial_types}_types", f"gamma{gamma}",
                                 f"spls_smi{smi:.5f}_beta{beta:.5f}.pkl")
        out_dir_path = os.path.join(cur_path, "SavedOutputs", "ReciprocalModel", "FullDataset", "connectomes",
                                    f"{num_artificial_types}_types", f"gamma{gamma}")
        if not os.path.exists(out_dir_path):
            try:
                os.mkdir(out_dir_path)
            except FileExistsError:
                pass
        run_reciprocity_dependence_model_distributed_full_dataset_single_dev_stage(spls_path, smi, beta, gamma,
                                                                                   out_dir_path,
                                                                                   num_artificial_types=num_artificial_types)


def train_single_epoch_consistent_birth_times():
    num_artificial_types = 8
    all_data_sets_path = os.path.join("InferredTypes", "ConsistentBirthTimes", "synapses_lists",
                                      f"{num_artificial_types}_types")
    dev_stages = [7]
    train_reciprocity_dependence_model_distributed_syn_list(num_artificial_types, dev_stages, all_data_sets_path,
                                                            "SingleDevStage")


def train_three_consistent_birth_times():
    num_artificial_types = 8
    all_data_sets_path = os.path.join("InferredTypes", "ConsistentBirthTimes", "synapses_lists",
                                      f"{num_artificial_types}_types")
    dev_stages = [1, 5, 7]
    train_reciprocity_dependence_model_distributed_syn_list(num_artificial_types, dev_stages, all_data_sets_path,
                                                            "ThreeDevStages")


def calc_custom_epochs_model_dyads_dists_across_splits_syn_list(dev_stages_str, developmental_ages):
    cur_path = os.getcwd()
    func_id = int(os.environ['LSB_JOBINDEX'])
    split = (func_id - 1) // 350 + 1
    age = (func_id % 350) * 10
    if age == 0:
        age = ADULT_WORM_AGE
    num_types = 8
    gamma = 7

    test_synapses_list_path = os.path.join(cur_path, "CElegansData", "InferredTypes", "ConsistentBirthTimes",
                                           "synapses_lists", f"{num_types}_types", f'split{split}', "test",
                                           "Dataset7.pkl")
    train_synapses_list_path = os.path.join(cur_path, "CElegansData", "InferredTypes", "ConsistentBirthTimes",
                                            "synapses_lists", f"{num_types}_types", f'split{split}', "train",
                                            "Dataset7.pkl")

    smi, beta, _ = find_max_likelihood_full_model(
        os.path.join(cur_path, "SavedOutputs", "ReciprocalModel", "BirthTimesConsistent", "likelihoods", dev_stages_str,
                     f"split{split}"))
    spls_file_name = f"spls_smi{smi:.5f}_beta{beta:.5f}.pkl"
    spls_path = os.path.join(cur_path, "SavedOutputs", "ReciprocalModel", "BirthTimesConsistent", "S+s", dev_stages_str,
                             f"split{split}", spls_file_name)
    with open(spls_path, 'rb') as f:
        spls = pickle.load(f)
    out_dir_path_dyads_dist_test = os.path.join(cur_path, "SavedOutputs", "ReciprocalModel", "BirthTimesConsistent",
                                                "dyads_distributions", dev_stages_str, 'TestSet', f'split{split}')
    if not os.path.exists(out_dir_path_dyads_dist_test):
        try:
            os.mkdir(out_dir_path_dyads_dist_test)
        except FileExistsError:
            pass
    out_dir_path_dyads_dist_test = os.path.join(out_dir_path_dyads_dist_test, f'{smi:.5f}')
    if not os.path.exists(out_dir_path_dyads_dist_test):
        try:
            os.mkdir(out_dir_path_dyads_dist_test)
        except FileExistsError:
            pass

    out_dir_path_dyads_dist_train = os.path.join(cur_path, "SavedOutputs", "ReciprocalModel", "BirthTimesConsistent",
                                                 "dyads_distributions", dev_stages_str, 'TrainSet', f'split{split}')
    if not os.path.exists(out_dir_path_dyads_dist_train):
        try:
            os.mkdir(out_dir_path_dyads_dist_train)
        except FileExistsError:
            pass
    out_dir_path_dyads_dist_train = os.path.join(out_dir_path_dyads_dist_train, f'{smi:.5f}')
    if not os.path.exists(out_dir_path_dyads_dist_train):
        try:
            os.mkdir(out_dir_path_dyads_dist_train)
        except FileExistsError:
            pass

    model_dyads_dist_train = calc_reciprocal_dependence_model_dyads_states_distribution_syn_list(spls, smi, beta, gamma,
                                                                                                 age,
                                                                                                 developmental_ages,
                                                                                                 train_synapses_list_path)
    with open(os.path.join(out_dir_path_dyads_dist_train, f"{age}.pkl"), 'wb') as f:
        pickle.dump(model_dyads_dist_train, f)

    model_dyads_dist_test = calc_reciprocal_dependence_model_dyads_states_distribution_syn_list(spls, smi, beta, gamma,
                                                                                                age,
                                                                                                developmental_ages,
                                                                                                test_synapses_list_path)
    with open(os.path.join(out_dir_path_dyads_dist_test, f"{age}.pkl"), 'wb') as f:
        pickle.dump(model_dyads_dist_test, f)


def calc_dyads_dists_single_epoch():
    dev_stages_str = "SingleDevStage"
    developmental_ages = {0: ADULT_WORM_AGE}
    calc_custom_epochs_model_dyads_dists_across_splits_syn_list(dev_stages_str, developmental_ages)


def calc_dyads_dists_three_epochs():
    dev_stages_str = "ThreeDevStages"
    developmental_ages = THREE_DEVELOPMENTAL_AGES_CONSISTENT
    calc_custom_epochs_model_dyads_dists_across_splits_syn_list(dev_stages_str, developmental_ages)


def main():
    train_reciprocity_dependence_model_distributed_single_stage_full_dataset()
    run_reciprocity_dependence_model_distributed_across_gammas_full_dataset_single_dev_stage(range(2, 10))

    train_single_epoch_consistent_birth_times()
    train_three_consistent_birth_times()

    save_max_like_params_per_split_single_epoch()
    save_max_like_params_per_split_three_epochs()

    calc_dyads_dists_single_epoch()
    calc_dyads_dists_three_epochs()


if __name__ == "__main__":
    main()
