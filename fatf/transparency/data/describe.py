"""
The :mod:`fatf.path.to.the.file.in.the.module` module implements description functionality for structured numpy arrays.
"""

# Author: Rafael Poyiadzi <rp13102@bristol.ac.uk>
# License: BSD 3 clause
import numpy as np
from fatf.utils.validation import check_array_type, is_2d_array
from typing import Optional, List, Union

def describe_numeric(series: np.ndarray):
    numeric_dict = {
            'count': series.shape[0],
            'mean': np.mean(series),
            'std': np.std(series),
            'max': np.max(series),
            'min': np.min(series),
             '25%': np.quantile(series, 0.25),
             '50%': np.quantile(series, 0.50),
             '75%': np.quantile(series, 0.75)
            }

    for key, value in numeric_dict.items():
        if key != 'count':
            numeric_dict[key] = value #format(value, '.2f')
    return numeric_dict

def describe_categorical(series: np.ndarray):
    unique, counter = np.unique(series, return_counts = True)
    top = np.argmax(counter)

    categorical_dict = {
            'count': series.shape[0],
            'count_unique': len(unique),
            'unique': unique,
            'most_common': unique[top],
            'most_common_count': counter[top],
            'hist': dict(zip(unique, counter))
            }

    return categorical_dict

def describe_dataset(dataset: np.ndarray, 
                     todescribe: Optional[List[str]] = None, 
                     condition: Optional[np.array] = None) -> dict:
    """Will provide a description of the desired fields of the dataset.

    Parameters
    ----------
    dataset : np.ndarray
        The dataset to be described.
    todescribe : Optional[list]
        A list of field names to be described. If none, then all will be
        described.
    condition : np.array
        Values used to provide conditional descriptions.
        
    Raises
    ------
    ValueError
        Dimensions of dataset and condition do not match.

    Returns
    -------
    If condition not provided:
        describe_dict : dict
            Dictionary of dictionaries. At first level keys correspond to fields
            that were described, and at second level you have key, value pairs
            for the statistics evaluated.
    Else:
        grand_dict : dict
            Dictionary of dictionaries of dictionaries. First level corresponds to 
            keys corresponding to the unique values in the condition array. The rest
            two levels correspond to a describe_dict.
    
    """
    check = is_2d_array(dataset)
    if not check:
        raise TypeError('Input should be 2-Dimensional')
    structured_bool = True
    if len(dataset.dtype) == 0:
        structured_bool = False
        
    numerical_fields, categorical_fields = check_array_type(dataset)
    if not todescribe:
        todescribe = numerical_fields.tolist() + categorical_fields.tolist()
    if condition is not None:
        values_set = list(set(condition))
        n_samples = condition.shape[0]
        if n_samples != dataset.shape[0]:
            raise ValueError('Dimension of condition does not match dimension of dataset')
            
        grand_dict = {}
        for value in values_set:
            mask = np.array(np.zeros(n_samples), dtype=bool)
            t = np.where(condition == value)[0]
            mask[t] = True
            describe_dict = {}

            for field_name in numerical_fields:
                if field_name in todescribe:
                    if structured_bool:
                        describe_dict[field_name] = describe_numeric(dataset[mask][field_name])
                    else:
                        describe_dict[field_name] = describe_numeric(dataset[mask][:, field_name])
            for field_name in categorical_fields:
                if field_name in todescribe:
                    if structured_bool:
                        describe_dict[field_name] = describe_categorical(dataset[mask][field_name])
                    else:
                        describe_dict[field_name] = describe_categorical(dataset[mask][:, field_name])
            grand_dict[value] = describe_dict
        return grand_dict
    else:
        describe_dict = {}
        for field_name in numerical_fields:
            if field_name in todescribe:
                if structured_bool:
                    describe_dict[field_name] = describe_numeric(dataset[field_name])
                else:
                    describe_dict[field_name] = describe_numeric(dataset[:, field_name])
        for field_name in categorical_fields:
            if field_name in todescribe:
                if structured_bool:
                    describe_dict[field_name] = describe_categorical(dataset[field_name])
                else:
                    describe_dict[field_name] = describe_categorical(dataset[:, field_name])
        return describe_dict

