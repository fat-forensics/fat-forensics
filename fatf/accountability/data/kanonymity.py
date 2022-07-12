# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 13:51:01 2018

@author: Rafael Poyiadzi
"""

from supplementary_funcs import check_input
from funcs import suppress, concatenate_sensitive_attributes

from kanonymity_funcs import cluster_and_generalize_kanonymity
from funcs import create_dataset

def kanonymize(data, treatments, lca_funcs, range_funcs, k):
    """ k-Anonymity function
    
    Description: A function that will produce 
    a k-anonymized dataset.
    
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
        k: The minimum size of the equivalence classes.
        
    Returns: A k-anonymized dataset.
    
    Raises:
        ValueError on 
        """
    quasi_identifiers = treatments['QI']
    attributes_to_suppress = treatments['I']
    sensitive_attributes = treatments['SA']
    
    check_input(data, quasi_identifiers, sensitive_attributes, attributes_to_suppress, k)
    
    data = suppress(data, attributes_to_suppress)
    if len(sensitive_attributes) > 1:
        data, sensitive_attributes = concatenate_sensitive_attributes(data, sensitive_attributes)
    
    features = quasi_identifiers
    data = cluster_and_generalize_kanonymity(data, features, lca_funcs, range_funcs, k)
    return data
 
dataset, treatments, lca_funcs, distance_funcs, range_funcs = create_dataset()

data = kanonymize(dataset, treatments, lca_funcs, range_funcs, 4)

d = sorted(data, key = lambda entry: entry['cluster'])
for row in d:
    print(row)
print(data.dtype)
