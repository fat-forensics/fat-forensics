# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 13:51:01 2018

@author: Rafael Poyiadzi
"""

from supp import input_dict
from supplementary_funcs import check_input
from funcs import suppress, concatenate_sensitive_attributes

from kanonymity_funcs import cluster_and_generalize_kanonymity

def kanonymize(data, quasi_identifiers, sensitive_attributes, attributes_to_suppress, k):
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
    
    check_input(data, quasi_identifiers, sensitive_attributes, attributes_to_suppress, k)
    
    data = suppress(data, attributes_to_suppress)
    
    if len(sensitive_attributes) > 1:
        data, sensitive_attributes = concatenate_sensitive_attributes(data, sensitive_attributes)
    
    data = cluster_and_generalize_kanonymity(data, k)
    return data
        
data = input_dict['data']
sensitive_attributes = list(input_dict['sensitive_attributes'])
quasi_identifiers = list(input_dict['quasi_identifiers'])

attributes_to_suppress=['name', 'email']
sensitive_attributes = ['diagnosis', 'gender']

data = kanonymize(data, quasi_identifiers, sensitive_attributes, attributes_to_suppress, 2)

d = sorted(data, key = lambda entry: entry['cluster'])
for row in d:
    print(row)
print(data.dtype)
