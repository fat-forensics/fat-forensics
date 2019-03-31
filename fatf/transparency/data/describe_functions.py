"""
Implements functions to describe numpy arrays.
"""
# Author: Rafael Poyiadzi <rp13102@bristol.ac.uk>
#         Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

from typing import Dict, List, Optional, Union

import numpy as np

import fatf.utils.array.tools as fuat
import fatf.utils.array.validation as fuav

from fatf.exceptions import IncorrectShapeError

__all__ = ['describe_categorical_array',
           'describe_numerical_array',
           'describe_array']  # yapf: disable


def describe_array(dataset: np.ndarray,
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
    if not fuav.is_2d_array(dataset):
        raise TypeError('Input should be 2-Dimensional')
    structured_bool = True
    if len(dataset.dtype) == 0:
        structured_bool = False

    numerical_fields, categorical_fields = fuat.indices_by_type(dataset)
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
                        describe_dict[field_name] = describe_numerical_array(dataset[mask][field_name])
                    else:
                        describe_dict[field_name] = describe_numerical_array(dataset[mask][:, field_name])
            for field_name in categorical_fields:
                if field_name in todescribe:
                    if structured_bool:
                        describe_dict[field_name] = describe_categorical_array(dataset[mask][field_name])
                    else:
                        describe_dict[field_name] = describe_categorical_array(dataset[mask][:, field_name])
            grand_dict[value] = describe_dict
        return grand_dict
    else:
        describe_dict = {}
        for field_name in numerical_fields:
            if field_name in todescribe:
                if structured_bool:
                    describe_dict[field_name] = describe_numerical_array(dataset[field_name])
                else:
                    describe_dict[field_name] = describe_numerical_array(dataset[:, field_name])
        for field_name in categorical_fields:
            if field_name in todescribe:
                if structured_bool:
                    describe_dict[field_name] = describe_categorical_array(dataset[field_name])
                else:
                    describe_dict[field_name] = describe_categorical_array(dataset[:, field_name])
        return describe_dict


def describe_numerical_array(
        array: Union[np.ndarray, np.void],
        skip_nans: bool = True) -> Dict[str, Union[int, float]]:
    """
    Describes a numerical numpy array with basic statistics.

    If the ``skip_nans`` parameter is set to ``True``, any ``numpy.nan``
    present in the input array is skipped for calculating the statistics.
    Otherwise, they are included, affecting most of the statistics and possibly
    equating them to ``numpy.nan``.

    The description outputted by this function is a dictionary with the
    following keys:

    ``count`` : integer
        The number of elements in the array.

    ``mean`` : float
        The *mean* (average) value of the array.

    ``std`` : float
        The *standard deviation* of the array.

    ``min`` : float
        The *minimum value* in the array.

    ``25%`` : float
        The *25 percentile* of the array.

    ``50%`` : float
        The *50 percentile* of the array, which is equivalent to its
        **median**.

    ``75%`` : float
        The *75 percentile* of the array.

    ``max`` : float
        The *maximum value* in the array.

    ``nan_count`` : integer
        The count of ``numpy.nan`` (not-a-number) values in the array.

    Parameters
    ----------
    array : Union[numpy.ndarray, numpy.void]
        An array for which description is desired.
    skip_nans : boolean, optional (default=True)
        If set to ``True``, ``numpy.nan``s present in the input array will be
        excluded while computing the statistics.

    Raises
    ------
    IncorrectShapeError
        The input array is not 1-dimensinoal.
    ValueError
        The input array is not purely numerical or it is empty.

    Returns
    -------
    numerical_description : Dict[str, Union[int, float]]
        A dictionary describing the numerical input array.
    """
    if not fuav.is_1d_like(array):
        raise IncorrectShapeError('The input array should be 1-dimensional.')

    classic_array = fuat.as_unstructured(array)
    assert len(classic_array.shape) == 1, '1D arrays only at this point.'

    if not classic_array.shape[0]:
        raise ValueError('The input array cannot be empty.')
    if not fuav.is_numerical_array(classic_array):
        raise ValueError('The input array should be purely numerical.')

    nan_indices = np.isnan(classic_array)
    n_elements = classic_array.shape[0]

    if skip_nans:
        classic_array = classic_array[~nan_indices]

    numerical_description = {
        'count': n_elements,
        'mean': np.mean(classic_array),
        'std': np.std(classic_array),
        'min': np.min(classic_array),
        '25%': np.quantile(classic_array, 0.25),
        '50%': np.quantile(classic_array, 0.50),
        '75%': np.quantile(classic_array, 0.75),
        'max': np.max(classic_array),
        'nan_count': nan_indices.sum()
    }

    return numerical_description


def describe_categorical_array(array: np.ndarray):
    unique, counter = np.unique(array, return_counts = True)
    top = np.argmax(counter)

    categorical_dict = {
        'count': array.shape[0],
        'count_unique': len(unique),
        'unique': unique,
        'most_common': unique[top],
        'most_common_count': counter[top],
        'hist': dict(zip(unique, counter))
    }

    return categorical_dict
