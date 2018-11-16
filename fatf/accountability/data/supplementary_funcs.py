# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 14:29:04 2018

@author: Rafael Poyiadzi
"""
from __future__ import division
import inspect
import numpy as np
from collections import Counter

def is_numeric(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def get_distr(inputlist):
    count = Counter(inputlist)
    s = sum(count.values())
    for key in count.keys():
        count[key] /= s
    return count

def get_emd_fordistrs(distr1, distr2):
    union_set = set(distr1.keys()).union(distr2.keys())
    emd = 0
    for item in union_set:
        try:
            val1 = distr1[item]
            try:
                val2 = distr2[item]
                emd += np.abs(val1 - val2)
            except:
                emd += val1 
        except:
            emd += distr2[item]
    return emd
            
def get_emd_forlists(list1, list2):
    count1 = get_distr(list1)
    count2 = get_distr(list2)
    
    return get_emd_fordistrs(count1, count2)

def encode_categorical(inputdata, alphabet, direction=1):
    if direction == 1:
        char_to_int = dict((c, i) for i, c in enumerate(alphabet))
        integer_encoded = [char_to_int[char] for char in inputdata]
        return integer_encoded
    else:
        int_to_char = dict((i, c) for i, c in enumerate(alphabet))
        onehot_encoded = list()
        for value in inputdata:
            letter = [0 for _ in range(len(alphabet))]
            letter[value] = 1
            onehot_encoded.append(letter)
        # invert encoding
        inverted = [int_to_char[np.argmax(onehot_encoded[i])] for i in range(len(onehot_encoded))]
        return inverted

def merge_values(data, attribute, values):
    column = data[attribute]
    newvalue = values[0] + values[1]
    for idx, val in enumerate(column):
        if val in values:
            data[attribute][idx] = newvalue
    return data

def view_counts(attribute):
    from collections import Counter
    num_codes = encode_categorical(attribute, set(attribute), 1)
    counts = Counter(num_codes)
    keys = counts.keys()
    values = counts.values()
    
    codes = encode_categorical(keys, set(attribute), 0)
    pl.bar(keys, values, color='lightblue', align='center')
    pl.xticks(keys, codes, rotation=45)
    #pl.xlim([-1,x_train.shape[1]])
    pl.tight_layout()
    pl.show() 
    
def check_input(data, quasi_identifiers, sensitive_attributes, attributes_to_suppress, k):
    """ Cheking whether the input is right.
    
    Description: Checks the input.
    
    Args: ...
    
    Raises:
        
    """
    
        
    if type(data) is not np.ndarray:
        raise TypeError('data needs to be of type "np.ndarray" ')
        
    if type(quasi_identifiers) is not list:
        raise TypeError('quasi_identifiers needs to be of type "list" ')
    
    if type(sensitive_attributes) is not list:
        raise TypeError('sensitive_attributes needs to be of type "list" ')
        
    if type(attributes_to_suppress) is not list:
        raise TypeError('attributes_to_suppress needs to be of type "list" ')

    for item in quasi_identifiers:
        if type(item) is not str:
            raise TypeError('quasi_identifiers should be of type "string" ')
    
    for item in sensitive_attributes:
        if type(item) is not str:
            raise TypeError('sensitive_attributes should be of type "string" ')
            
    for item in attributes_to_suppress:
        if type(item) is not str:
            raise TypeError('attributes_to_suppress should be of type "string" ')
            
    if not sensitive_attributes:
        raise ValueError('No Sensitive Attributes Provided')
    
    if not quasi_identifiers:
        raise ValueError('No Quasi-Identifiers Provided')
    
    if not is_numeric(k):
        raise ValueError('k needs to be numeric')
    
    if k < 1:
        raise ValueError('k needs to be positive')

    if data.shape[0] < k:
        raise ValueError('The size of the dataset needs to be larger than k')
    
        
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)
    
    if calframe[1][3] == 'ldiversify':
        if len(sensitive_attributes) == 1:
            if len(set(data[sensitive_attributes[0]])) < k:
                raise ValueError('Not enough distinct sensitive attributes')
        else:
            newcolumn = np.array(list(map(lambda x: x[0] + '-' + x[1], data[sensitive_attributes])))
            if len(set(newcolumn)) < k:
                raise ValueError('Not enough distinct sensitive attributes')
                
    attributes = data.dtype.fields.keys()
    for attr in sensitive_attributes:
        if attr not in attributes:
            raise NameError(str(attr) + ' in Sensitive Attributes, not found in the dataset')
    
    for attr in quasi_identifiers:
        if attr not in attributes:
            raise NameError(str(attr) + ' in Quasi-Identifiers, not found in the dataset')
        
    for attr in attributes_to_suppress:
        if attr not in attributes:
            raise NameError(str(attr) + ' in attributes to suppresss, not found in the dataset')
        