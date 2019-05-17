"""
Convert dataset into binary dataset.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
# License: new BSD

import warnings

from typing import List, Optional, Union

import numpy as np

from fatf.exceptions import IncorrectShapeError

import fatf.utils.array.tools as fuat
import fatf.utils.array.validation as fuav

__all__ = ['similarity_binary_mask']


def _validate_input(dataset: np.ndarray,
                    data_row: np.ndarray):
    """
    Validates the input to :func:`fatf.utils.data.similarity_binary_mask.
    similarity_binary_mask`.

    This function checks if ``dataset`` is a valid 2-dimensional array and
    if ``data_row`` is a valid 1-dimensional array. Also checks if they have
    valid dtypes and whether to dtypes are equivalent.

    For additional parameters, attributes, warnings and exceptions raised by
    this class please see the  documentation of the function
    :func:`fatf.utils.data.similarity_binary_mask.similarity_binary_mask`.

    Returns
    -------
    is_valid : boolean
        ``True`` if input is valid, ``False`` otherwise.
    """
    is_valid = False

    if not fuav.is_2d_array(dataset):
        raise IncorrectShapeError('The input dataset must be a '
                                  '2-dimensional numpy array.')

    if not fuav.is_base_array(dataset):
        raise TypeError('The input dataset must be of a base type.')

    if not fuav.is_1d_like(data_row):
        raise IncorrectShapeError('The data_row must either be a 1-dimensional '
                                  'numpy array or numpy void object for '
                                  'structured rows.')

    are_similar = fuav.are_similar_dtype_arrays(
                dataset, np.array([data_row]), strict_comparison=True)
    if not are_similar:
        raise TypeError('The dtype of the data_row is different to '
                        'the dtype of the dataset provided.')

    if not fuav.is_structured_array(dataset):
                if data_row.shape[0] != dataset.shape[1]:
                    raise IncorrectShapeError('The data_row must contain the '
                                              'same number of features as the '
                                              'dataset provided.')

    is_valid = True
    return is_valid


def similarity_binary_mask(dataset: np.ndarray,
                           data_row: np.ndarray,):
    """
    Converts dataset values to `1` if the same as a row and `0` if not.
    
    Coverts dataset into a binary dataset where values are `1` if the
    ``dataset`` value is equal to ``data_row`` value and `0` if not. Also will
    output feature names for the binary features. Also converts data types to
    integers.

    
    Parameters
    ----------
    dataset : numpy.ndarray
        A 2-dimensional numpy array with a dataset to be turned into binary
        dataset.
    data_row : numpy.ndarray
        A 1-dimensional array containing feature values that will be compared 
        against ``dataset`` parameter.

    Raises
    ------
    IncorrectShapeError
        dataset is not a 2-dimensional array or data_row is not a 
        1-dimensional array or data_row and dataset do not have the same 
        number of features in them.
    TypeError
        If dataset is not a base dtype or data_row is not a base dtype or 
        dataset and data_row have different dtypes.

    Returns
    -------
    binary_dataset: numpy.ndarray
        Binary data.
    """
    assert _validate_input(dataset, data_row), 'Input is not valid.'
    is_structured = fuav.is_structured_array(dataset)

    if is_structured:
        dtypes = [(name, np.int32) for name in dataset.dtype.names]
    else:
        dtypes = np.int32

    binary_dataset = np.zeros_like(dataset, dtype=dtypes)

    indices = dataset.dtype.names if is_structured else \
        list(range(dataset.shape[1]))

    for index in indices:
        if is_structured:
            column = dataset[index]
            binary_dataset[index] = column == data_row[index]
        else:
            column = dataset[:, index]
            binary_dataset[:, index] = column == data_row[index]

    return binary_dataset
