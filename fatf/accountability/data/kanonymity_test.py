# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 13:51:01 2018

@author: Rafael Poyiadzi
"""

from supp import input_dict
from supplementary_funcs import check_input
from funcs import suppress, concatenate_sensitive_attributes

from kanonymity_funcs import cluster_and_generalize_kanonymity
from testing import create_dataset

def kanonymize(data, treatments, lca_funcs, k):
    """ k-Anonymity function
    
    Description: A function that will produce 
    a k-anonymized dataset.
    
    Args: 
        data: The dataset to be anonymized, Structured Numpy Array.
        sensitive_attributes: The list of attributes to be protected.
        quasi_identifiers: The list of attributes that could serve
                           as identifiers.
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
    
    data = cluster_and_generalize_kanonymity(data, lca_funcs, k)
    return data
 
dataset, treatments, lca_funcs, distance_funcs = \
    create_dataset()

data = kanonymize(dataset, treatments, lca_funcs, 2)

d = sorted(data, key = lambda entry: entry['cluster'])
for row in d:
    print(row)
print(data.dtype)
