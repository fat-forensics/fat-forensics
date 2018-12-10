"""
The :mod:`fatf.utils.validation` module holds all functions that are used
for object validation across FAT-Forensics.
"""

# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: BSD 3 clause

import inspect
import numpy as np
import warnings

from typing import Tuple

from fatf.exceptions import (CustomValueError,
                             IncorrectShapeException)

# Boolean, (signed) byte -- Boolean, unsigned integer, (signed) integer,
# floating-point and complex-floating point.
_NUMPY_NUMERICAL_KINDS = set('?buifc')

def is_numerical_dtype(dtype: np.dtype) -> bool:
    """Determine whether a numpy dtype object is of numerical kind.

    Check whether the dtype is of one of the following (numerical) types:
    Boolean, (signed) byte -- Boolean, unsigned integer, (signed) integer,
    floating-point or complex-floating point.

    Parameters
    ----------
    array : np.dtype
        The dtype to be checked.

    Raises
    ------
    CustomValueError
        The input is not a numpy dtype object.

    Returns
    -------
    is_numerical : bool
        True if the dtype is of a numerical type, False otherwise.
    """
    if not isinstance(dtype, np.dtype):
        raise CustomValueError('The input should be a numpy dtype object.')

    is_numerical = True if dtype.kind in _NUMPY_NUMERICAL_KINDS else False

    return is_numerical

def is_numerical_array(array: np.ndarray) -> bool:
    """Determine whether a numpy array-like object has a numerical data type.

    Check whether the array is of one of the following (numerical) types:
    Boolean, (signed) byte -- Boolean, unsigned integer, (signed) integer,
    floating-point or complex-floating point.

    Parameters
    ----------
    array : np.ndarray
        The array to be checked.

    Raises
    ------
    CustomValueError
        The input array is not a numpy array-like object.

    Returns
    -------
    is_numerical : bool
        True if the array has a numerical data type, False otherwise..
    """
    if not isinstance(array, np.ndarray):
        raise CustomValueError('The input should be a numpy array-like.')

    is_numerical = is_numerical_dtype(array.dtype)

    return is_numerical

def is_1d_array(array: np.ndarray) -> bool:
    """Determine whether a numpy array-like object has 1 dimensions.

    Parameters
    ----------
    array : np.ndarray
        The array to be checked.

    Raises
    ------
    CustomValueError
        The input array is not a numpy array-like object.

    Returns
    -------
    is_1d : bool
        True if the array is 2-dimensional, False otherwise.
    """
    if not isinstance(array, np.ndarray):
        raise CustomValueError('The input should be a numpy array-like.')
    
    if is_structured(array):
        is_1d = True if array.shape[0] == 1 else False
    else:
        if len(array.dtype) == 0:
            is_1d = True if len(array.shape) == 1 else False
        else:
            is_1d = True if len(array.shape) == 0 else False
    
    return is_1d

def is_2d_array(array: np.ndarray) -> bool:
    """Determine whether a numpy array-like object has 2 dimensions.

    Parameters
    ----------
    array : np.ndarray
        The array to be checked.

    Raises
    ------
    CustomValueError
        The input array is not a numpy array-like object.

    Returns
    -------
    is_2d : bool
        True if the array is 2-dimensional, False otherwise.
    """
    if not isinstance(array, np.ndarray):
        raise CustomValueError('The input should be a numpy array-like.')

    if is_numerical_array(array):
        is_2d = True if len(array.shape) == 2 else False
    else:
        if len(array.dtype) == 0:
            is_2d = True if len(array.shape) == 2 else False
        else:
            is_2d = True if len(array.shape) == 1 else False

    return is_2d

def is_structured(array: np.ndarray) -> bool:
    """Determine whether a numpy array-like object is a structured array
    i.e. contains more than one datatype

    Args
    ----
    array : np.ndarray
        The array to be checked.

    Returns
    ----
    is_structured : bool
        True if the array is a structured array, False otherwise
    """
    return True if len(array.dtype) !=0 else False

def check_array_type(array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Check whether a numpy array is purely numerical or a structured array
    and return two lists: one with numerical indices and the other with
    categorical indices.

    This function inspects the dtype of the input array and decides on type of
    each column in the array.

    Parameters
    ----------
    distance : np.ndarray
        A numpy array to be checked (it has to be a 2 dimensional array).

    Raises
    ------
    CustomValueError
        The input array is not a numpy array-like object.
    IncorrectShapeException
        The input array is not 2-dimensional.

    Returns
    -------
    numerical_indices : np.ndarray
        A numpy array containing indices of the numerical columns of the
        original array.
    categorical_indices : np.ndarray
        A numpy array containing indices of the categorical columns of the
        original array.
    """
    if not isinstance(array, np.ndarray):
        raise CustomValueError('The input should be a numpy array-like.')

    if not is_2d_array(array):
        raise IncorrectShapeException('The input array should be '
                                      '2-dimensional.')
    if not is_structured(array):
        if is_numerical_array(array):
            numerical_indices = np.array(range(array.shape[1]))
            categorical_indices = np.empty((0,), dtype='i8')
        else:
            numerical_indices = np.empty((0,), dtype='i8')
            categorical_indices = np.array(range(array.shape[1]))
    else:
        if len(array.dtype) == 0:
            # If it's not numerical and it's of a single type, all the columns
            # are categorical.
            numerical_indices = np.empty((0,))
            categorical_indices = np.array(range(array.shape[1]))
        else:
            numerical_indices_list = []
            categorical_indices_list = []

            for column_name in array.dtype.names:
                column_dtype = array.dtype[column_name]
                if is_numerical_dtype(column_dtype):
                    numerical_indices_list.append(column_name)
                else:
                    categorical_indices_list.append(column_name)

            numerical_indices = np.array(numerical_indices_list)
            categorical_indices = np.array(categorical_indices_list)

    return numerical_indices, categorical_indices

def check_indices(array: np.ndarray,
                  indices: np.ndarray) -> np.ndarray:
    """Check if indices are valid and return a list of indices that are not found
    in the array

    Args
    ----
    array : np.ndarray
        The array to be checked
    indices : np.ndarray
        1-D array of indices corresponding to features in array
    
    Returns
    ----
    invalid_indices : np.ndarray
        Array of indices that are not found in array
    """
    if not is_1d_array(indices):
        invalid_ind = indices
    else:
        numerical_indices, categorical_indices = check_array_type(array)
        valid_indices = np.hstack([numerical_indices, categorical_indices])
        #TODO: np.in1d raises internal numpy FutureWarning, not sure how to get around it
        invalid_ind = indices[~np.in1d(indices, valid_indices)]
    return invalid_ind

def check_valid_indices(array: np.array, 
                        indices: np.array) -> bool:
    """Check whether indices given in indices are valid indices for
    array.

    Args
    ----
    array : np.array
        The array to be checked.
    categorical_indices : np.array
        1-D array of indices corresponding to features in array

    Returns
    ----
    is_valid : bool
        A Boolean variable that indicates whether the entries of indices
        are valid indices in array 
    """
    is_valid = True
    invalid_indices = check_indices(array, indices)
    if not np.array_equal(invalid_indices, np.array([], dtype=indices.dtype)):
        is_valid = False
    return is_valid

def check_model_functionality(model_object: object,
                              require_probabilities: bool = False,
                              verbose: bool = False) -> bool:
    """Check whether a model class has all the required methods with the correct
    number of parameters (excluding self): __init__ (at least 0), fit (at
    least 2), predict (at least 1) and, if required, predict_proba (at least 1).

    Parameters
    ----------
    model_object : object
        A Python object that represents a model.
    require_probabilities : bool, optional
        A Boolean parameter that indicates whether the model object should
        contain a predict_proba method. Defaults to False.
    verbose : bool, optional
        A Boolean parameter that indicates whether the function should print
        warnings. Defaults to False.

    Warns
    -----
    Warning
        The model object lacks required functionality.

    Returns
    -------
    is_functional : bool
        A Boolean variable that indicates whether the model object has all the
        desired functionality.
    """
    is_functional = True

    methods = {
        'fit': 2,
        'predict': 1
    }
    if require_probabilities:
        methods['predict_proba'] = 1

    message_strings = []
    for method in methods:
        if not hasattr(model_object, method):
            is_functional = False
            message_strings.append(
                    'The model class is missing \'{}\' method.'.format(method)
                )
        else:
            method_object = getattr(model_object, method)
            required_param_n = 0
            params = inspect.signature(method_object).parameters
            for param in params:
                if params[param].default is inspect._empty:
                    required_param_n += 1
            if required_param_n != methods[method]:
                is_functional = False
                message_strings.append(
                        ('The \'{}\' method of the class has incorrect number '
                         '({}) of the required parameters. It needs to have '
                         'exactly {} required parameters. Try using optional '
                         'parameters if you require more functionality.'
                        ).format(method, required_param_n, methods[method])
                    )

    if verbose and not is_functional:
        message = '\n'.join(message_strings)
        warnings.warn(message, category=Warning)

    return is_functional
