import os
import pickle
import networkx as nx
from copy import deepcopy
import numpy as np

from train_c_elegans_independent_model import calc_spl_per_type_single_developmental_stage, explore_model_likelihood
from CElegansNeuronsAdder import CElegansNeuronsAdder
from c_elegans_constants import SINGLE_DEVELOPMENTAL_AGE, ADULT_WORM_AGE
from c_elegans_independent_model_training import calc_model_adj_mat


def _create_connectome_copy_from_template_adj_mat(template, adj_mat, node_list):
    new = deepcopy(template)
    num_nodes = len(node_list)
    edges = [(node_list[i], node_list[j], {
        'length': np.linalg.norm(template.nodes[node_list[j]]['coords'] - template.nodes[node_list[i]]['coords']),
        'birth time': max(template.nodes[node_list[j]]['birth_time'], template.nodes[node_list[i]]['birth_time'])}) for
             i in range(num_nodes) for j in range(num_nodes) if adj_mat[i, j]]
    new.add_edges_from(edges)
    return new


def generate_overlap_data(num_types):
    data_dir_path = f"CElegansData\\InferredTypes\\connectomes\\{num_types}_types"
    with open(os.path.join(data_dir_path, "Dataset7.pkl"), 'rb') as f:
        worm_7 = pickle.load(f)
    with open(os.path.join(data_dir_path, "Dataset8.pkl"), 'rb') as f:
        worm_8 = pickle.load(f)
    with open('CElegansData\\worm_atlas_sub_connectome_chemical_no_autosynapses_subtypes.pkl', 'rb') as f:
        worm_atlas = pickle.load(f)

    connectome_template = deepcopy(worm_7)
    connectome_template.clear_edges()
    node_list = sorted(list(connectome_template.nodes))
    worm_7_adj_mat = nx.to_numpy_array(worm_7, nodelist=node_list)
    worm_8_adj_mat = nx.to_numpy_array(worm_8, nodelist=node_list)
    worm_atlas_adj_mat = nx.to_numpy_array(worm_atlas, nodelist=node_list)

    worm_atlas_num_types = _create_connectome_copy_from_template_adj_mat(connectome_template, worm_atlas_adj_mat,
                                                                         node_list)
    with open(os.path.join(data_dir_path, "worm_atlas.pkl"), 'wb') as f:
        pickle.dump(worm_atlas_num_types, f)

    worms_7_8_overlap_mat = worm_7_adj_mat * worm_8_adj_mat
    worms_7_8_overlap = _create_connectome_copy_from_template_adj_mat(connectome_template, worms_7_8_overlap_mat,
                                                                      node_list)
    with open(os.path.join(data_dir_path, "worms_7_8_overlap.pkl"), 'wb') as f:
        pickle.dump(worms_7_8_overlap, f)

    worms_7_atlas_overlap_mat = worm_7_adj_mat * worm_atlas_adj_mat
    worms_7_atlas_overlap = _create_connectome_copy_from_template_adj_mat(connectome_template,
                                                                          worms_7_atlas_overlap_mat,
                                                                          node_list)
    with open(os.path.join(data_dir_path, "worms_7_atlas_overlap.pkl"), 'wb') as f:
        pickle.dump(worms_7_atlas_overlap, f)

    worms_8_atlas_overlap_mat = worm_8_adj_mat * worm_atlas_adj_mat
    worms_8_atlas_overlap = _create_connectome_copy_from_template_adj_mat(connectome_template,
                                                                          worms_8_atlas_overlap_mat,
                                                                          node_list)
    with open(os.path.join(data_dir_path, "worms_8_atlas_overlap.pkl"), 'wb') as f:
        pickle.dump(worms_8_atlas_overlap, f)

    worms_7_8_atlas_overlap_mat = worm_7_adj_mat * worm_8_adj_mat * worm_atlas_adj_mat
    worms_7_8_atlas_overlap = _create_connectome_copy_from_template_adj_mat(connectome_template,
                                                                            worms_7_8_atlas_overlap_mat,
                                                                            node_list)
    with open(os.path.join(data_dir_path, "worms_7_8_atlas_overlap.pkl"), 'wb') as f:
        pickle.dump(worms_7_8_atlas_overlap, f)


def train_single_stage_inferred_types(connectome_name, num_types,
                                      params_file_name="spl_per_type_params_low_smi_low_beta.pkl"):
    cur_path = os.getcwd()
    connectome_relative_path = os.path.join("InferredTypes", "connectomes",
                                            f"{num_types}_types", connectome_name)

    spls_single_path = os.path.join(cur_path, "SavedOutputs", "BackboneModel", "S+s",
                                    f"{num_types}_types", connectome_name)
    os.makedirs(spls_single_path, exist_ok=True)
    spls, smi, beta = calc_spl_per_type_single_developmental_stage(
        os.path.join(cur_path, "ParamFiles", params_file_name),
        spls_single_path,
        CElegansNeuronsAdder.ARTIFICIAL_TYPES, connectome_file_name=connectome_relative_path,
        num_artificial_types=num_types)
    likelihood_single_path = os.path.join(cur_path, "SavedOutputs", "BackboneModel", "likelihoods",
                                          f"{num_types}_types", connectome_name)
    os.makedirs(likelihood_single_path, exist_ok=True)
    explore_model_likelihood(params_file_name, SINGLE_DEVELOPMENTAL_AGE,
                             likelihood_single_path,
                             spls_single_path, connectome_relative_path, batch_size=1)
    av_mats_single_path = os.path.join(cur_path, "SavedOutputs", "BackboneModel", "average_adj_mats",
                                       f"{num_types}_types", connectome_name)
    os.makedirs(av_mats_single_path, exist_ok=True)
    average_mat = calc_model_adj_mat(spls, smi, beta, ADULT_WORM_AGE, SINGLE_DEVELOPMENTAL_AGE,
                                     os.path.join(cur_path, "CElegansData", connectome_relative_path))
    with open(os.path.join(av_mats_single_path, f"spls_smi{smi:.5f}_beta{beta:.5f}.pkl"), 'wb') as f:
        pickle.dump(average_mat, f)


def main():
    num_types = 8
    # generate_overlap_data(num_types)
    # func_id = int(os.environ['LSB_JOBINDEX']) - 1
    # data_set_names = ["Dataset7.pkl", "Dataset8.pkl", "worm_atlas.pkl", "worms_7_8_overlap.pkl",
    #                   "worms_7_atlas_overlap.pkl", "worms_8_atlas_overlap.pkl", "worms_7_8_atlas_overlap.pkl"]
    # data_set_idx = func_id // 1600
    # data_set_name = data_set_names[data_set_idx]
    # train_single_stage_inferred_types(data_set_name, num_types)

    # from er_block_model import greedy_type_aggregation
    # worm_8_path = 'CElegansData\\InferredTypes\\connectomes\\8_types\\Dataset8.pkl'
    # with open(worm_8_path, 'rb') as f:
    #     worm_8 = pickle.load(f)
    # worm_8_adj_mat = nx.to_numpy_array(worm_8, nodelist=sorted(list(worm_8.nodes)))
    # greedy_type_aggregation(worm_8_adj_mat, "D:\OrenRichter\\temp\\worm_8_greedy_type_aggregation")

    func_id = int(os.environ['LSB_JOBINDEX']) - 1
    data_set_names = ["Dataset8_worm_8_types.pkl", "worm_atlas_worm_atlas_types.pkl"]
    data_set_idx = func_id // 1600
    data_set_name = data_set_names[data_set_idx]
    train_single_stage_inferred_types(data_set_name, num_types)



if __name__ == "__main__":
    main()
