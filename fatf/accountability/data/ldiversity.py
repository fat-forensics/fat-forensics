# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 13:51:01 2018

@author: Rafael Poyiadzi
"""

from supplementary_funcs import check_input
from funcs import suppress, concatenate_sensitive_attributes

from ldiversity_funcs import cluster_and_generalize_ldiversity
from funcs import create_dataset

def ldiversity(data, treatments, lca_funcs, range_funcs, l):
    """ l-Diversity function
    
    Description: A function that will produce 
    a l-diversified dataset -- Each equivalence class will hold at least
    l distinct values of the provided 'Sensitive Attributes'
    
    Args: 
        data: The dataset to be anonymized, Structured Numpy Array.
        treatments: The dictionary with the 3 types of treatment:
                    'QI' - Quasi-Identifier, 
                    'I' - Identifier and
                    'SA' - Sensitive Attribute as keys. For values
                    it should have the relevant features.
        lca_funcs: List of user provided functions - one for each
                    'QI' to be used to compute the lower common ancestor
                    for a group of points.
        range_funcs: List of user provided functions - one for each 'QI' 
                    to be used to calculate the range of a feature in
                    a group of points. For continuous variables, it is the 
                    range of the variable in the group (MAX - MIN) and for 
                    categorical it is the height of the tree.
        l: The number of distinct values in each equivalence class, with respect
            to the sensitive attribute.
        
    Returns: An l-diversified dataset.
    
    Raises:
        ValueError on 
        """
      
    quasi_identifiers = treatments['QI']
    attributes_to_suppress = treatments['I']
    sensitive_attributes = treatments['SA']
    # Checking the input
    check_input(data, quasi_identifiers, sensitive_attributes, attributes_to_suppress, l)
    # Suppresing sensitive attributes
    data = suppress(data, attributes_to_suppress)
    
    # Combine the sensitive attributes if more than one
    # for now, only two
    if len(sensitive_attributes) > 1:
        data, sensitive_attributes = concatenate_sensitive_attributes(data, sensitive_attributes)
    
    features = quasi_identifiers
    data = cluster_and_generalize_ldiversity(data, features, lca_funcs, range_funcs, sensitive_attributes[0], l)
    return data
        
dataset, treatments, lca_funcs, distance_funcs, range_funcs = create_dataset()

data = ldiversity(dataset, treatments, lca_funcs, range_funcs, 3)

d = sorted(data, key = lambda entry: entry['cluster'])
for row in d:
    print(row)
print(data.dtype)

