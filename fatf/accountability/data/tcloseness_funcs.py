# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 14:26:05 2018

@author: Rafael Poyiadzi
"""
import numpy as np
from collections import Counter

from ldiversity_funcs import clustering_for_ldiversity
from funcs import assign_extras, get_equivalence_classes, initialize_clustering
from information_metrics import get_information_loss, get_total_information_loss, filter_dataset, get_feature_ranges
from supplementary_funcs import get_distr, get_emd_forlists, get_emd_fordistrs

def check_tcloseness(attribute, cluster_assignments, base_counts, t):
    """
    Check whether the t-closeness condition is satisfied, for all the clusters.
    """
    satisfied = True
    emds = []
    for cluster in set(cluster_assignments):
        filtered_attribute = attribute[np.where(cluster_assignments == cluster)[0]]
        emd = get_emd_fordistrs(get_distr(filtered_attribute), base_counts)
        emds.append((cluster, emd))
        if emd >= t:
            satisfied = False
    return satisfied, emds
    
def merge_clusters(clusters_tobe_merged, cluster_assignments):
    newcluster = int(np.max(cluster_assignments) + 1)
    for idx, val in enumerate(cluster_assignments):
        if int(val) in list(map(int, clusters_tobe_merged)):
            cluster_assignments[idx] = newcluster
    return cluster_assignments

def cluster_and_generalize_tcloseness(data, features, lca_funcs, range_funcs, sensitive_attribute, t):
    """ Cluster and generalize for t-closeness.
    
    Description: Function that will call clustering and generalization
                functions for t-closeness. The algorithm will first form
                clusters of similar instances of size at least 'starting_size'
                and then combine clusters until t-closeness is satisfied. 
                
                The clustering part follows by 'clustering_for_ldiversity', 
                which could change to 'clustering_for_kanonymity'.
                
                Guarantees t-closeness.
    
    Args: 
        data: Structured Numpy Array dataset to be anonymized.
        features: List of attributes to be used when clustering.
        lca_funcs: List of user provided functions - one for each
                    'QI' to be used to compute the lower common ancestor
                    for a group of points.
        range_funcs: List of user provided functions - one for each 'QI' 
                    to be used to calculate the range of a feature in
                    a group of points. For continuous variables, it is the 
                    range of the variable in the group (MAX - MIN) and for 
                    categorical it is the height of the tree.
        sensitive_attribute: The name (string) of the attribute to be protected.
        t: The minimum allowed distance between the distribution of the 
            sensitive attribute in each equivalence class and the distribution
            of the attribute in the full dataset, as calculated by the 
            Earth Movers Distance (EMD).
        
    Returns: A dataset that satisfies t-closeness.
    
    Raises:
        NA 
        """
    base_counts = Counter(data[sensitive_attribute])

    starting_size = len(set(data[sensitive_attribute]))

    # cluster the data according to l-diversity rule.
    cluster_assignments = clustering_for_ldiversity(data, features, range_funcs, sensitive_attribute, starting_size)
                
    base_distr = get_distr(base_counts)
    
    # If the formed clusters satisfy the t-closeness condition,
    # then proceed to form the equivalence classes.
    satisfied, emds = check_tcloseness(data[sensitive_attribute], cluster_assignments, base_distr, t)
    SA = data[sensitive_attribute]
    
    # While the t-closeness condition has not been met, start merging clusters
    # until it is satisfied.
    # ** EMD(P, Q) <= max{EMD(P0, Q), EMD(P1, Q)} **
    # where P0 and P1 are the distributions of the sensitive attribute in two
    # clusters and P is the distribution of sensitive attribute when the
    # two clusters are merged. Q is the distribution of the sensitive attribute
    # on the full dataset.
    
    # The procedure is to find the cluster that has the highest EMD to Q
    # and then find a second cluster which is the closest to the first
    # and merge them.
    
    while not satisfied:
        clusters = [item[0] for item in emds]
        furthest_cluster = clusters[np.argmax([item[1] for item in emds])]
        list1 = SA[np.where(cluster_assignments == furthest_cluster)]
        scores = []
        for cluster in set(cluster_assignments):
            if cluster == furthest_cluster:
                scores.append((cluster, 1000))
                continue
            list2 = SA[np.where(cluster_assignments == cluster)[0]]
            emd = get_emd_forlists(list1, list2)
            scores.append((cluster, emd))
        emds_tocluster = np.array([item[1] for item in scores])
        closest_cluster = clusters[emds_tocluster.argsort()[0]]
        clusters_tobe_merged = [furthest_cluster, closest_cluster]
        cluster_assignments = merge_clusters(clusters_tobe_merged, cluster_assignments)
        satisfied, emds = check_tcloseness(data[sensitive_attribute], cluster_assignments, base_distr, t)
        
    data = get_equivalence_classes(data, features, lca_funcs, cluster_assignments)
    
    return data, emds

def swap_instances_(SA, base_distr, cluster_assignments, cluster_counter, t):
    """ Swaps instances between a cluster and the pool of unlabelled instances.
    
    Description: Find the best possible swap of instances between the given
                cluster and the pool of unlabelled instances. It iterates
                over all possible swaps to find the one that reduces EMD the most.
                   
    Args: 
        data: Structured Numpy Array dataset.
        sensitive_attribute: String for the name of the attribute to be
                            protected.
        base_distr: The distribution of the sensitive_attribute in the whole
                    dataset.
        cluster_assignments: List of cluster assignments
        t: The maximum allowed Earth Mover's Distance (EMD) distance allowed
            between each equivalence class' distribution of sensitive_attribute
            and the overall distribution of the sensitive_attribute.
        
    Returns: cluster_assignments: List with updated cluster_assignments, with a lower
                                t-distance.
    
    Raises:
        NA 
        """
    scores = []
    for internal_idx, internal_val in enumerate(cluster_assignments):
        if internal_val == cluster_counter:
            for external_idx, external_val in enumerate(cluster_assignments):
                if external_val == -1:
                    cluster_assignments[internal_idx] = -1
                    cluster_assignments[external_idx] = cluster_counter
                    new_distr = get_distr(SA[np.where(cluster_assignments == cluster_counter)[0]])
                    new_emd = get_emd_fordistrs(new_distr, base_distr)
                    
                    scores.append((internal_idx, external_idx, new_emd))
                    
                    cluster_assignments[internal_idx] = cluster_counter
                    cluster_assignments[external_idx] = -1
    best = np.argmin([item[2] for item in scores])
    internal, external = scores[best][0], scores[best][1]
    cluster_assignments[internal] = -1
    cluster_assignments[external] = cluster_counter
    return cluster_assignments

def swap_instances(data, features, range_funcs, sensitive_attribute, base_distr, cluster_assignments, t):
    """ Swaps instances between a cluster and the pool of unlabelled instances.
    
    Description: Find the best possible swap of instances between the given
                cluster and the pool of unlabelled instances. It first selects 
                the instance who reduces the information loss within the cluster
                and then finds the best replacement with respect to reducing
                the EMD to the base distribution.
                   
    Args: 
        data: Structured Numpy Array dataset.
        sensitive_attribute: String for the name of the attribute to be
                            protected.
        base_distr: The distribution of the sensitive_attribute in the whole
                    dataset.
        cluster_assignments: List of cluster assignments
        t: The maximum allowed Earth Mover's Distance (EMD) distance allowed
            between each equivalence class' distribution of sensitive_attribute
            and the overall distribution of the sensitive_attribute.
        
    Returns: cluster_assignments: List with updated cluster_assignments, with a 
                                lower EMD to the base distribution.
    
    Raises:
        NA 
        """
    cluster_counter = int(max(cluster_assignments))
    SA = data[sensitive_attribute]
    scores = []
    
    full_dataset_ranges = get_feature_ranges(data, features, range_funcs)
    
    for idx, val in enumerate(cluster_assignments):
        if val == cluster_counter:
            cluster_assignments[idx] = -1
            filtered_dataset = filter_dataset(data, cluster_assignments, [cluster_counter])
            scores.append((idx, get_information_loss(filtered_dataset, features, full_dataset_ranges, range_funcs)))
            cluster_assignments[idx] = cluster_counter
            
    best_internal = scores[np.argmax([item[1] for item in scores])][0]
    
    scores = []
    for external_idx, external_val in enumerate(cluster_assignments):
        if external_val == -1:
            cluster_assignments[best_internal] = -1
            cluster_assignments[external_idx] = cluster_counter
            new_distr = get_distr(SA[np.where(cluster_assignments == cluster_counter)[0]])
            new_emd = get_emd_fordistrs(new_distr, base_distr)
            
            scores.append((external_idx, new_emd))
            
            cluster_assignments[best_internal] = cluster_counter
            cluster_assignments[external_idx] = -1
    best = np.argmin([item[1] for item in scores])
    best_external = scores[best][0]
    
    cluster_assignments[best_internal] = -1
    cluster_assignments[best_external] = cluster_counter
    return cluster_assignments

def cluster_and_generalize_tcloseness_2(data, features, lca_funcs, range_funcs, sensitive_attribute, t):
    """ Cluster and generalize for t-diversity.
    
    Description: Function that will call clustering and generalization
                functions for t-closeness. The algorithm will first form
                one cluster and then keep swapping instances with the
                set of unlabelled instances until the first cluster satisfies
                t-closeness. It then continues by forming the second cluster
                and so on and so forth.
                
                Does not guarantee t-closeness in its current form.
    
    Args: 
        data: Structured Numpy Array dataset to be anonymized.
        features: List of attributes to be used when clustering.
        lca_funcs: List of user provided functions - one for each
                    'QI' to be used to compute the lower common ancestor
                    for a group of points.
        range_funcs: List of user provided functions - one for each 'QI' 
                    to be used to calculate the range of a feature in
                    a group of points. For continuous variables, it is the 
                    range of the variable in the group (MAX - MIN) and for 
                    categorical it is the height of the tree.
        sensitive_attribute: The name (string) of the attribute to be protected.
        t: The minimum allowed distance between the distribution of the 
            sensitive attribute in each equivalence class and the distribution
            of the attribute in the full dataset, as calculated by the 
            Earth Movers Distance (EMD).
        
    Returns: A dataset that satisfies t-closeness.
    
    Raises:
        NA 
        """
    base_counts = Counter(data[sensitive_attribute])
    base_distr = get_distr(base_counts)
    
    starting_size = len(set(data[sensitive_attribute]))
    k = starting_size
    
    n = data.shape[0]
    cluster_assignments = initialize_clustering(n)
    
    SA = data[sensitive_attribute] 
    cluster_counter = 0
    positions_of_unclustered = np.where(cluster_assignments == -1)[0]

    while len(positions_of_unclustered) > k:
        while len(np.where(cluster_assignments == cluster_counter)[0]) < k:
            scores = []
            for idx, val in enumerate(cluster_assignments):
                if val == -1:
                    cluster_assignments[idx] = cluster_counter
                    scores.append((idx, get_total_information_loss(data, features, range_funcs, cluster_assignments)))
                    cluster_assignments[idx] = -1
            best = scores[np.argmin(np.array([item[1] for item in scores]))][0]
            cluster_assignments[best] = cluster_counter
        new_distr = get_distr(SA[np.where(cluster_assignments == cluster_counter)[0]])    
        emd = get_emd_fordistrs(new_distr, base_distr)
        satisfied = emd <= t

        if not satisfied:
            cluster_assignments = swap_instances(data, features, range_funcs, sensitive_attribute, base_distr, cluster_assignments, t)
            #cluster_assignments = swap_instances_(SA, base_distr, cluster_assignments, cluster_counter, t)
        positions_of_unclustered = np.where(cluster_assignments == -1)[0]
        cluster_counter += 1
        
    satisfied, emds = check_tcloseness(data[sensitive_attribute], cluster_assignments, base_distr, t)
    cluster_assignments = assign_extras(data, features, range_funcs, cluster_assignments)

    data = get_equivalence_classes(data, features, lca_funcs, cluster_assignments)
    satisfied, emds = check_tcloseness(data[sensitive_attribute], cluster_assignments, base_distr, t)

    return data, emds
        