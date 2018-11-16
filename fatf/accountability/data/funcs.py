from __future__ import division
import numpy as np
from numpy.lib import recfunctions as rfn

from information_metrics import get_total_information_loss
from testing import get_data

def suppress(dataset, attributes_to_suppress):
    """ Suppression function
    
    Description: Will suppress all the values in 'dataset' under,
                the fields provided in 'attributes_to_suppress'.
    
    Args: 
        dataset: Structured Numpy Array dataset to be anonymized.
        attributes_to_suppress: List of strings representing attributes,
                                to be suppressed.
        
    Returns: The dataset provided but with the fields specified suppressed.
    
    Raises:
        ValueError when no attributes to be suppressed are provided.
        TypeError when the attributes to be provided are not strings.
        """
        
    if not attributes_to_suppress:
        raise ValueError('No attributes_to_suppress have been provided')
    
    for item in attributes_to_suppress:
        if type(item) is not str:
            raise TypeError('attributes_to_suppress should be of type "str" ')
            
    for attr in attributes_to_suppress:
        dataset[attr] = '*'
    return dataset

def concatenate_sensitive_attributes(data, sensitive_attributes, newfield_name = None, newfield_type = '<U30'):
    """ Combines sensitive attributes in the case of them being more than one.
    
    Description: Cross product between the sensitive attributes.
    
    Args: 
        dataset: Structured Numpy Array dataset to be anonymized.
        sensitive_attributes: List of strings representing the attributes to be combined.
        newfield_name: Option to provide the name of the new field. Default
                        is the concatenation of the two.
        newfield_type: Option to provide the dtype of the newfield. Default
                        is '<U30'.
                        
    Returns: The dataset provided but with the previous fields removed,
            and a new field - their combinarion - being included.
    
    Raises:
        NA
        """
    newcolumn = np.array(list(map(lambda x: x[0] + '-' + x[1], data[sensitive_attributes])))
    if not newfield_name:
        newfield_name = sensitive_attributes[0] + '-' + sensitive_attributes[1]
    else:
        if type(newfield_name) != str:
            newfield_name = str(newfield_name)
            
    newdataset = rfn.append_fields(data, newfield_name, newcolumn, dtypes=newfield_type).data
    newdataset = rfn.drop_fields(newdataset, sensitive_attributes)
    return newdataset, [newfield_name]

def get_lowest_common_ancestor(data, features, lca_funcs):
    """ Computes the lowest common ancestor (LCA) for the given dataset.
    
    Description: A function that will compute the LCA for the given dataset
    
    Args: 
        data: Structured Numpy Array for which the LCA is desired.
        features: List of attributes to be used when clustering.
        lca_funcs: List of user provided functions - one for each
                    'QI' to be used to compute the lower common ancestor
                    for a group of points.
        
    Returns: The lowest common ancestor of the dataset, across all features.
    
    Raises:
        NA 
        """
    equivalence_class = {}
    for attr_name, attr_lca_func in lca_funcs:
        if attr_name in features:
            equivalence_class[attr_name] = attr_lca_func(data[attr_name])

    return equivalence_class

def create_dataset():
    list_of_dictionaries = get_data()

    desired_keys = ['name',
                    'data',
                    'treatment',
                    'lca_func',
                    'range_func',
                    'distance_func']
    
    dts = []
    treatments = {
                'I': [],
                'SA': [],
                'QI': []
                }
    distance_funcs = []
    lca_funcs = []
    data = []
    
    range_funcs = []
    
    for dictionary in list_of_dictionaries:
        current_dictionary_keys = dictionary.keys()
        for key in desired_keys:
            if key not in current_dictionary_keys:
                raise ValueError('One of the provided dictionaries does not have the key: ' + str(key))
        
        field_name = dictionary['name']
        field_col = dictionary['data']
        if type(field_col) != np.ndarray:
            raise TypeError(str(field_name) + ' data should be of type numpy.ndarray.')
        
        data.append(field_col)
        
        dts.append((field_name, field_col.dtype))
        distance_funcs.append((field_name, dictionary['distance_func']))
        
        current_field_lca_func = dictionary['lca_func']
        lca_funcs.append((field_name, current_field_lca_func))
        
        current_field_range_func = dictionary['range_func']
        range_funcs.append((field_name, current_field_range_func))

        field_treatment = dictionary['treatment']
        
        if field_treatment == 'I':
            treatments['I'].append(field_name)
        elif field_treatment == 'SA':
            treatments['SA'].append(field_name)
        elif field_treatment == 'QI':
            treatments['QI'].append(field_name)
            if not current_field_lca_func:
                raise ValueError(str(field_name) + ' field requires an LCA function.')
            if not current_field_range_func:
                raise ValueError(str(field_name) + ' field requires a range function.')
        else:
            raise ValueError('Unknown treatment')
            
    N = data[0].shape[0]
    if not np.all(column.shape[0] == N for column in data):
        raise ValueError('Data provided is of different length.')
        
    dataset = np.array([item for item in zip(*data)], dtype=dts)
    return dataset, treatments, lca_funcs, distance_funcs, range_funcs

def get_equivalence_classes(data, features, lca_funcs, cluster_assignments):
    """ Function that will form the equivalence class (EQ) for each cluster.
    
    Description: For each of the clusters and for each of the features provided,
                will compute the EQ using the lca_funcs provided by the user.
    
    Args: 
        data: Structured Numpy Array dataset to be anonymized.
        features: List of features that will be used to calculate information loss.
        lca_funcs: List of functions, one per feature, to be used to compute the
                    lowest common ancestor for each cluster for each feature.
        clustering_assignments: List of clustering assignments
        
    Returns: The Equivalence class of each cluster.
    
    Raises:
        NA 
    """
    ###
    data = change_numerical_to_str(data)
    ###
    for cluster in set(cluster_assignments):
        dataset = data[cluster_assignments == cluster]
        equivalence_class = get_lowest_common_ancestor(dataset, features, lca_funcs)
        for key, value in equivalence_class.items():
            dataset[key] = str(value)
        data[cluster_assignments == cluster] = dataset

    newdata = rfn.append_fields(data, 'cluster', cluster_assignments).data
    return newdata

def change_numerical_to_str(data):
    dt = []
    for attr, attr_type in data.dtype.fields.items():
        if attr_type[0] == 'int32':
            dt.append((attr, np.dtype('<U6')))
        else:
            dt.append((attr, attr_type[0]))  
    return data.astype(dt) 

def initialize_clustering(n):
    cluster_assignments = np.array(-np.ones(n))
    idx = np.random.randint(n, size=1)    
    cluster_assignments[idx] = 0
    return cluster_assignments

def assign_extras(data, features, range_extras, cluster_assignments):
    """ Function to alocate the remaining datapoints to existing clusters.
    
    Description: After the main clusters have been formed, this function will
                alocate the remaining datapoints to them. For example, in 
                k-anonymity this function will be called when the remaining instances
                are fewer than k. For l-diversity it will be called when the 
                remaining distinct elements for the Sensitive Attribute are 
                fewer than l.
    
    Args: 
        dataset: Structured Numpy Array dataset to be anonymized.
        features: List of features that will be used to calculate information loss.
        range_funcs: List of functions, one per feature, to be used to calculate
                    the range of each feature. For continuous, it is of the form
                    'MAX - MIN', while for categorical it is the height of the
                    tree.
        clustering_assignments: List of clustering assignments
        
    Returns: Finalized cluster assignments.
    
    Raises:
        NA 
    """
    cluster_counter = int(max(cluster_assignments))
    while sum(cluster_assignments == -1) > 0:
        for idx, val in enumerate(cluster_assignments):
            if val != -1:
                continue
            scores = []
            for cluster in range(cluster_counter+1):
                cluster_assignments[idx] = cluster
                scores.append((idx, get_total_information_loss(data, features, range_extras, cluster_assignments)))
                cluster_assignments[idx] = -1
            best = scores[np.argmin(np.array([item[1] for item in scores]))][0]
            cluster_assignments[best] = cluster
    return cluster_assignments




 



        
        
        
        
        