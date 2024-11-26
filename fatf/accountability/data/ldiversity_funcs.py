# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 14:15:22 2018

@author: Rafael Poyiadzi
"""

import numpy as np
from funcs import initialize_clustering, assign_extras, get_equivalence_classes
from information_metrics import get_total_information_loss

def cluster_and_generalize_ldiversity(data, features, lca_funcs, range_funcs, sensitive_attribute, l):
    """ Cluster and generalize for l-diversity.
    
    Description: Function that will call clustering and generalization
                functions for l-diversity.
    
    Args: 
        data: Structured Numpy Array dataset to be anonymized.
        sensitive_attribute: String for the name of the attribute to be
                            protected.
        l: The number of distinct entries in each equivalence class, with
            respect to the sensitive_attribute
        
    Returns: An l-diversified dataset.
    
    Raises:
        NA 
        """
    cluster_assignments = clustering_for_ldiversity(data, features, range_funcs, sensitive_attribute, l)
    data = get_equivalence_classes(data, features, lca_funcs, cluster_assignments)

    return data

def clustering_for_ldiversity(data, features, range_funcs, sensitive_attribute, l):
    """ Cluster for l-diversity.
    
    Description: Function that will cluster the data in clusters where 
                each cluster has at least l distinct entries in the 
                sensitive_attribute. The algorithm is an extension
                of the algorithm presented in the paper 
                'Efficient k-Anonymization Using Clustering Techniques'
                that deals with k-anonymity.
    
    Args: 
        data: Structured Numpy Array dataset to be anonymized.
        sensitive_attribute: String for the name of the attribute to be 
                            protected
        l: The minimum number of distinct entries in each cluster,
            for the sensitive_attribute
        
    Returns: Clustering assignments for the data provided.
    
    Raises:
        NA 
    """    
    n = data.shape[0]
    cluster_assignments = initialize_clustering(n)
    
    SA = data[sensitive_attribute] 
    positions_of_unclustered = np.where(cluster_assignments == -1)[0]
    cluster_counter = 0
    
    # The algorithm will first form clusters where each one has at least
    # l distinct entries in the sensitive_attribute feature, and then
    # assign the extras to the existing clusters.
    
    # A cluster assignment of -1 implies not assigned to a cluster yet.
    while len(set(SA[positions_of_unclustered])) >= l:
        while sum(cluster_assignments == cluster_counter) < l:
            scores = []
            positions_of_cluster = np.where(cluster_assignments == cluster_counter)[0]
            filt_attribute = SA[positions_of_cluster]
            for idx, val in enumerate(cluster_assignments):
                if (val == -1 and SA[idx] not in set(filt_attribute)):
                    cluster_assignments[idx] = cluster_counter
                    scores.append((idx, get_total_information_loss(data, features, range_funcs, cluster_assignments)))
                    cluster_assignments[idx] = -1
            best = scores[np.argmin(np.array([item[1] for item in scores]))][0]
            cluster_assignments[best] = cluster_counter

        cluster_counter += 1
        positions_of_unclustered = np.where(cluster_assignments == -1)[0]
        
    cluster_assignments = assign_extras(data, features, range_funcs, cluster_assignments)

    return cluster_assignments
