# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 14:01:22 2018

@author: Rafael Poyiadzi
"""
def get_total_information_loss(dataset, features, range_funcs, cluster_assignments):
    """ Calculates the total information loss for the current clustering.
    
    Description: Function that will calculate the total information loss,
                for the current clustering, based on the 'features' provided.
    
    Args: 
        dataset: Structured Numpy Array dataset to be anonymized.
        features: List of features that will be used to calculate information loss.
        range_funcs: List of functions, one per feature, to be used to calculate
                    the range of each feature. For continuous, it is of the form
                    'MAX - MIN', while for categorical it is the height of the
                    tree.
        clustering_assignments: List of clustering assignments
        
    Returns: The total information loss for the current clustering.
    
    Raises:
        NA 
    """
    unique_clusters = list(set(cluster_assignments))
    full_dataset_ranges = get_feature_ranges(dataset, features, range_funcs)

    total_information_loss = 0
    for cluster in unique_clusters:
        if cluster == -1:
            continue
        filtered_dataset = filter_dataset(dataset, cluster_assignments, [cluster])
        total_information_loss += get_information_loss(filtered_dataset, features, full_dataset_ranges, range_funcs)
    return total_information_loss

def get_information_loss(dataset, features, full_dataset_ranges, range_funcs):
    """ Calculates the information loss for the current cluster.
    
    Description: Function that will calculate the total information loss,
                for the current clustering, based on the 'features' provided. 
                For continuous variables, it is the range of the variable in the 
                cluster (MAX - MIN) divided by the range in the full dataset, 
                multiplied by the size of the cluster. For categorical it is 
                the ratio of the heights of the trees, multiplied by the size
                of the cluster.
    
    Args: 
        dataset: Structured Numpy Array dataset to be anonymized.
        features: List of features that will be used to calculate information loss.
        range_funcs: List of functions, one per feature, to be used to calculate
                    the range of each feature. For continuous, it is of the form
                    'MAX - MIN', while for categorical it is the height of the
                    tree.
        clustering_assignments: List of clustering assignments
        
    Returns: The information loss for the current cluster.
    
    Raises:
        NA 
    """
    dataset_length = dataset.shape[0]
    information_loss = 0
    for attr_name, attr_range_func in range_funcs:
        if attr_name in features:
            attr_range = attr_range_func(dataset[attr_name])            
            information_loss += attr_range / full_dataset_ranges[attr_name]
    return int(dataset_length*information_loss)

def get_feature_ranges(dataset, features, range_funcs):
    """
    Applied the range_funcs on each feature.
    """
    ranges = {}
    for attr_name, attr_range_func in range_funcs:
        if attr_name in features:
            ranges[attr_name] = attr_range_func(dataset[attr_name])
    return ranges
 
def filter_dataset(dataset, cluster_assignments, filters):
    """
    Returns the entries of the dataset for the specific clusters.
    """
    filter_list = []
    for value in cluster_assignments:
        if value in filters:
            filter_list.append(True)
        else:
            filter_list.append(False)
    return dataset[filter_list]


  