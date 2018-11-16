# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 13:56:34 2018

@author: Rafael Poyiadzi
"""
import numpy as np
from funcs import initialize_clustering, assign_extras, get_equivalence_classes
from information_metrics import get_total_information_loss

def cluster_and_generalize_kanonymity(data, features, lca_funcs, range_funcs, k): 
    """ Cluster and generalize for k-anonymity.
    
    Description: Function that will call clustering and generalization
                functions for k-anonymity.
    
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
        k: The minimum size of the equivalence classes.
        
    Returns: A k-anonymized dataset.
    
    Raises:
        NA 
        """
    cluster_assignments = clustering_for_kanonymity(data, features, range_funcs, k)    
    data = get_equivalence_classes(data, features, lca_funcs, cluster_assignments)
    
    return data

def clustering_for_kanonymity(data, features, range_funcs, k):
    """ Cluster for k-anonymity.
    
    Description: Function that will cluster the data in clusters of 
                size at least k. The algorithm follows from paper 
                'Efficient k-Anonymization Using Clustering Techniques'.
    
    Args: 
        data: Structured Numpy Array dataset to be anonymized.
        features: List of attributes to be used when clustering.
        range_funcs: List of user provided functions - one for each 'QI' 
                    to be used to calculate the range of a feature in
                    a group of points. For continuous variables, it is the 
                    range of the variable in the group (MAX - MIN) and for 
                    categorical it is the height of the tree.
        k: The minimum size of the equivalence classes.
        
    Returns: Clustering assignments for the data provided, where each
            cluster is at least of size k.
    
    Raises:
        NA 
    """
    n = data.shape[0]
    cluster_assignments = initialize_clustering(n)
    cluster_counter = 0
    # The algorithm will first form clusters of size k with an objective,
    # of minimizing Information Loss, and then assign the leftover to the 
    # existing clusters.
    
    # A cluster assignment of -1 implies not assigned to a cluster yet.
    while sum(cluster_assignments == -1) > k:
        while sum(cluster_assignments == cluster_counter) < k:
            scores = []
            for idx, val in enumerate(cluster_assignments):
                if val == -1:
                    cluster_assignments[idx] = cluster_counter
                    scores.append((idx, get_total_information_loss(data, features, range_funcs, cluster_assignments)))
                    cluster_assignments[idx] = -1
            best = scores[np.argmin(np.array([item[1] for item in scores]))][0]
            cluster_assignments[best] = cluster_counter
        cluster_counter += 1
     
    # Stop forming new clusters, since now we have fewer than k elements left,
    # and start adding them to existing clusters, by trying to minimize
    # information loss again.
    cluster_assignments = assign_extras(data, features, range_funcs, cluster_assignments)

    return cluster_assignments
