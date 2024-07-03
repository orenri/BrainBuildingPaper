import numpy as np


# calculates the joint distribution induces by 2 partitions (clusterings) of a set -
# the probability to be in cluster i in the first partition and in cluster j in the second is
# #(elements in cluster_1_i and cluster_2_j) / #(elements in the set)
def generate_cluster_intersection_joint_prob_map(group_list, clusters_list_1, clusters_list_2):
    clusters_intersection_sizes = np.zeros((len(clusters_list_1), len(clusters_list_2)))
    for element in group_list:
        for i in range(len(clusters_list_1)):
            if element in clusters_list_1[i]:
                cluster_1 = i
                break
        for j in range(len(clusters_list_2)):
            if element in clusters_list_2[j]:
                cluster_2 = j
                break
        clusters_intersection_sizes[cluster_1, cluster_2] += 1

    set_size = len(group_list)
    induced_prob = clusters_intersection_sizes / set_size
    return induced_prob
