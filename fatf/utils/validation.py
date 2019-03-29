"""
Holds functions responsible for objects validation across FAT-Forensics.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import inspect
import warnings

from typing import Tuple, Union

import numpy as np

from fatf.exceptions import IncorrectShapeError

__all__ = ['is_numerical_dtype',
           'is_textual_dtype',
           'is_base_dtype',
           'are_similar_dtypes',
           'are_similar_dtype_arrays',
           'is_numerical_array',
           'is_textual_array',
           'is_base_array',
           'is_1d_array',
           'is_2d_array',
           'is_structured_row',
           'is_1d_like',
           'is_structured_array',
           'indices_by_type',
           'get_invalid_indices',
           'are_indices_valid',
           'check_model_functionality']  # yapf: disable

# Unsigned byte, Boolean, (signed) byte -- Boolean, unsigned integer,
# (signed) integer, floating-point and complex-floating point.
_NUMPY_NUMERICAL_KINDS = set('B?buifc')
# Unicode string
_NUMPY_TEXTUAL_KINDS = set('U')
# Zero-terminated bytes
_NUMPY_TEXTUAL_KINDS_UNSUPPORTED = set('Sa')
# O, M, m and V are considered complex objects
_NUMPY_BASE_KINDS = set('?buifcBSaU')


def is_numerical_dtype(dtype: np.dtype) -> bool:
    """
    Determines whether a numpy dtype object is of numerical type.

    Checks whether the ``dtype`` is of one of the following (numerical) types:
    unsigned byte, boolean, (signed) byte -- boolean, unsigned integer,
    (signed) integer, floating-point or complex-floating point.

    Parameters
    ----------
    dtype : numpy.dtype
        The dtype to be checked.

    Raises
    ------
    TypeError
        The input is not a numpy's dtype object.
    ValueError
        The dtype is structured -- this function only accepts plane dtypes.

    Returns
    -------
    is_numerical : boolean
        True if the dtype is of a numerical type, False otherwise.
    """
    if not isinstance(dtype, np.dtype):
        raise TypeError('The input should be a numpy dtype object.')

    # If the dtype is complex
    if dtype.names is not None:
        raise ValueError('The numpy dtype object is structured. '
                         'Only base dtype are allowed.')

    is_numerical = dtype.kind in _NUMPY_NUMERICAL_KINDS

    return is_numerical


def is_textual_dtype(dtype: np.dtype) -> bool:
    """
    Determines whether a numpy dtype object is of textual type.

    Checks whether the ``dtype`` is a unicode string type (textual). The
    zero-terminated bytes type is unsupported and not considered a textual
    type.

    Parameters
    ----------
    dtype : numpy.dtype
        The dtype to be checked.

    Raises
    ------
    TypeError
        The input is not a numpy's dtype object.
    ValueError
        The dtype is structured -- this function only accepts plane dtypes.

    Returns
    -------
    is_textual : boolean
        True if the dtype is of a textual type, False otherwise.
    """
    if not isinstance(dtype, np.dtype):
        raise TypeError('The input should be a numpy dtype object.')

    # If the dtype is complex
    if dtype.names is not None:
        raise ValueError('The numpy dtype object is structured. '
                         'Only base dtype are allowed.')

    if dtype.kind in _NUMPY_TEXTUAL_KINDS_UNSUPPORTED:
        warnings.warn('Zero-terminated bytes type is not supported and is not '
                      'considered to be a textual type. Please use any other '
                      'textual type.', category=UserWarning)
        is_textual = False
    else:
        is_textual = dtype.kind in _NUMPY_TEXTUAL_KINDS

    return is_textual


def is_base_dtype(dtype: np.dtype) -> bool:
    """
    Determines whether a numpy dtype object is one of base types.

    Checks whether the ``dtype`` is of any type but ``numpy.void`` --
    this usually happens when a numpy array holds objects instead of base
    type entities.

    Parameters
    ----------
    dtype : numpy.dtype
        The dtype to be checked.

    Raises
    ------
    TypeError
        The input is not a numpy's dtype object.
    ValueError
        The dtype is structured -- this function only accepts plane dtypes.

    Returns
    -------
    is_basic : boolean
        True if the dtype is of a base type, False otherwise.
    """
    if not isinstance(dtype, np.dtype):
        raise TypeError('The input should be a numpy dtype object.')

    # If the dtype is complex
    if dtype.names is not None:
        raise ValueError('The numpy dtype object is structured. '
                         'Only base dtype are allowed.')

    is_basic = dtype.kind in _NUMPY_BASE_KINDS

    return is_basic


def are_similar_dtypes(dtype_a: np.dtype,
                       dtype_b: np.dtype,
                       strict_comparison: bool = False) -> bool:
    """
    Checks whether two numpy dtypes are similar.

    If ``strict_comparison`` is set to True the both dtypes have to be exactly
    the same. Otherwise, if both are either numerical or textual dtypes, they
    are considered similar.

    Parameters
    ----------
    dtype_a : numpy.dtype
        The first dtype to be compared.
    dtype_b : numpy.dtype
        The first dtype to be compared.
    strict_comparison : boolean, optional (default=False)
        When set to True the dtypes have to match exactly. Otherwise, if both
        are either numerical or textual dtypes, they are considered similar.

    Raises
    ------
    TypeError
        Either of the inputs is not a numpy's dtype object.
    ValueError
        Either of the input dtypes is structured -- this function only accepts
        plane dtypes.

    Returns
    -------
    are_similar : boolean
        True if both dtypes are similar, False otherwise.
    """
    if not isinstance(dtype_a, np.dtype):
        raise TypeError('dtype_a should be a numpy dtype object.')
    if not isinstance(dtype_b, np.dtype):
        raise TypeError('dtype_b should be a numpy dtype object.')

    if dtype_a.names is not None:
        raise ValueError('The dtype_a is a structured numpy dtype object. '
                         'Only base dtype are allowed.')
    if dtype_b.names is not None:
        raise ValueError('The dtype_b is a structured numpy dtype object. '
                         'Only base dtype are allowed.')

    are_similar = False
    if strict_comparison:
        are_similar = dtype_a == dtype_b
    else:
        if ((is_numerical_dtype(dtype_a) and is_numerical_dtype(dtype_b))
                or (is_textual_dtype(dtype_a) and is_textual_dtype(dtype_b))):
            are_similar = True
        else:
            are_similar = dtype_a == dtype_b
    return are_similar


def are_similar_dtype_arrays(array_a: np.ndarray,
                             array_b: np.ndarray,
                             strict_comparison: bool = False) -> bool:
    """
    Determines whether two numpy array-like object have a similar data type.

    If ``strict_comparison`` is set to True the dtypes of both arrays have to
    be exactly the same. Otherwise, if both their dtypes are either numerical
    or textual dtypes, they are considered similar.

    If one of the arrays is a structured array and the other one is a classic
    numpy array the function returns False.

    Parameters
    ----------
    array_a : numpy.ndarray
        The first array to be checked.
    array_b : numpy.ndarray
        The second array to be checked.
    strict_comparison : boolean, optional (default=False)
        When set to True the dtypes have to match exactly. Otherwise, if both
        are either numerical or textual dtypes, they are considered similar.

    Raises
    ------
    TypeError
        Either of the inputs is not a numpy array-like object.

    Returns
    -------
    are_similar : boolean
        True if both arrays have a similar dtype, False otherwise.
    """
    if not isinstance(array_a, np.ndarray):
        raise TypeError('array_a should be a numpy array-like object.')
    if not isinstance(array_b, np.ndarray):
        raise TypeError('array_b should be a numpy array-like object.')

    is_a_structured = is_structured_array(array_a)
    is_b_structured = is_structured_array(array_b)
    if is_a_structured and is_b_structured:
        are_similar = True
        if len(array_a.dtype) != len(array_b.dtype):
            are_similar = False

        # Check names and types.
        if are_similar:
            for i in range(len(array_a.dtype)):
                are_similar = array_a.dtype.names[i] == array_b.dtype.names[i]
                if not are_similar:
                    break

                are_similar = are_similar_dtypes(
                    array_a.dtype[i], array_b.dtype[i], strict_comparison)
                if not are_similar:
                    break
    elif not is_a_structured and not is_b_structured:
        are_similar = are_similar_dtypes(
            array_a.dtype, array_b.dtype, strict_comparison)
    else:
        are_similar = False

    return are_similar


def is_numerical_array(array: np.ndarray) -> bool:
    """
    Determines whether a numpy array-like object has a numerical data type.

    Checks whether the ``array`` is of one of the following (numerical) types:
    boolean, (signed) byte -- boolean, unsigned integer, (signed) integer,
    floating-point or complex-floating point.

    Parameters
    ----------
    array : numpy.ndarray
        The array to be checked.

    Raises
    ------
    TypeError
        The input array is not a numpy array-like object.

    Returns
    -------
    is_numerical : boolean
        True if the array has a numerical data type, False otherwise.
    """
    if not isinstance(array, np.ndarray):
        raise TypeError('The input should be a numpy array-like object.')

    if is_structured_array(array):
        is_numerical = True
        for i in range(len(array.dtype)):
            if not is_numerical_dtype(array.dtype[i]):
                is_numerical = False
                break
    else:
        is_numerical = is_numerical_dtype(array.dtype)

    return is_numerical


def is_textual_array(array: np.ndarray) -> bool:
    """
    Determines whether a numpy array-like object has a textual data type.

    Checks whether the ``array`` is a unicode string type (textual). The
    zero-terminated bytes type is unsupported and not considered a textual
    type.

    Parameters
    ----------
    array : numpy.ndarray
        The array to be checked.

    Raises
    ------
    TypeError
        The input array is not a numpy array-like object.

    Returns
    -------
    is_textual : boolean
        True if the array has a textual data type, False otherwise.
    """
    if not isinstance(array, np.ndarray):
        raise TypeError('The input should be a numpy array-like object.')

    if is_structured_array(array):
        is_textual = True
        for i in range(len(array.dtype)):
            if not is_textual_dtype(array.dtype[i]):
                is_textual = False
                break
    else:
        is_textual = is_textual_dtype(array.dtype)

    return is_textual


def is_base_array(array: np.ndarray) -> bool:
    """
    Determines whether a numpy array-like object holds base data types.

    Checks whether the ``array`` is of any type but ``numpy.void`` --
    this usually happens when a numpy array holds objects instead of base
    type entities.

    Parameters
    ----------
    array : numpy.ndarray
        The array to be checked.

    Raises
    ------
    TypeError
        The input array is not a numpy array-like object.

    Returns
    -------
    is_basic : boolean
        True if the array is of a base data type, False otherwise.
    """
    if not isinstance(array, np.ndarray):
        raise TypeError('The input should be a numpy array-like object.')

    if is_structured_array(array):
        is_basic = True
        for i in range(len(array.dtype)):
            if not is_base_dtype(array.dtype[i]):
                is_basic = False
                break
    else:
        is_basic = is_base_dtype(array.dtype)

    return is_basic


def is_1d_array(array: np.ndarray) -> bool:
    """
    Determines whether a numpy array-like object is 1-dimensional.

    Parameters
    ----------
    array : numpy.ndarray
        The array to be checked.

    Raises
    ------
    TypeError
        The input array is not a numpy array-like object.

    Warns
    -----
    UserWarning
        The input array is 1-dimensional but its components are 1D structured.

    Returns
    -------
    is_1d : boolean
        True if the array is 1-dimensional, False otherwise.
    """
    if not isinstance(array, np.ndarray):
        raise TypeError('The input should be a numpy array-like.')

    if is_structured_array(array):
        is_1d = False
        if len(array.dtype) == 1 and len(array.shape) == 1:
            message = ('Structured (pseudo) 1-dimensional arrays are not '
                       'acceptable. A 1-dimensional structured numpy array '
                       'can be expressed as a classic numpy array with a '
                       'desired type.')
            warnings.warn(message, category=UserWarning)
    else:
        is_1d = len(array.shape) == 1

    return is_1d


def is_2d_array(array: np.ndarray) -> bool:
    """
    Determines whether a numpy array-like object has 2 dimensions.

    Parameters
    ----------
    array : numpy.ndarray
        The array to be checked.

    Raises
    ------
    TypeError
        The input array is not a numpy array-like object.

    Warns
    -----
    UserWarning
        The input array is 2-dimensional but its components are 1D structured.

    Returns
    -------
    is_2d : boolean
        True if the array is 2-dimensional, False otherwise.
    """
    if not isinstance(array, np.ndarray):
        raise TypeError('The input should be a numpy array-like.')

    if is_structured_array(array):
        if len(array.shape) == 2 and len(array.dtype) == 1:
            is_2d = False
            message = ('2-dimensional arrays with 1D structured elements are '
                       'not acceptable. Such a numpy array can be expressed '
                       'as a classic 2D numpy array with a desired type.')
            warnings.warn(message, category=UserWarning)
        elif len(array.shape) == 1 and len(array.dtype) > 0:
            is_2d = True
            for name in array.dtype.names:
                if array.dtype[name].ndim:
                    # This is a complex (multi-dimensional) embedded dtype
                    is_2d = False
                    break
        else:
            is_2d = False
    else:
        is_2d = len(array.shape) == 2

    return is_2d


def is_structured_row(structured_row: np.void) -> bool:
    """
    Determines whether the input is a structured numpy array's row object.

    Parameters
    ----------
    structured_row : numpy.void
        The object to be checked.

    Raises
    ------
    TypeError
        The input is not a structured numpy array's row object.

    Returns
    -------
    is_structured_row : boolean
        True if the input is array is a structured numpy array's row object,
        False otherwise.
    """
    if not isinstance(structured_row, np.void):
        raise TypeError('The input should be a row of a structured numpy '
                        'array (numpy.void type).')

    return len(structured_row.dtype) != 0


def is_1d_like(oned_like_object: Union[np.ndarray, np.void]) -> bool:
    """
    Checks if the input is either a 1D numpy array or a structured numpy row.

    Parameters
    ----------
    oned_like_object : Union[numpy.ndarray, numpy.void]
        The object to be checked.

    Raises
    ------
    TypeError
        The input is neither a numpy ndarray -- array-like object -- nor a
        numpy void -- a row of a structured numpy array.

    Returns
    -------
    is_1d_like : boolean
        True if the input is either a 1-dimensional numpy array or a row of a
        structured numpy array, False otherwise.
    """
    is_1d_like = False
    if isinstance(oned_like_object, np.void):
        is_1d_like = is_structured_row(oned_like_object)
    elif isinstance(oned_like_object, np.ndarray):
        is_1d_like = is_1d_array(oned_like_object)
    else:
        raise TypeError('The input should either be a numpy array-like object '
                        '(numpy.ndarray) or a row of a structured numpy array '
                        '(numpy.void).')

    return is_1d_like


def is_structured_array(array: np.ndarray) -> bool:
    """
    Determines whether a numpy array-like object is a structured array.

    Parameters
    ----------
    array : numpy.ndarray
        The array to be checked.

    Raises
    ------
    TypeError
        The input array is not a numpy array-like object.

    Returns
    -------
    is_structured : boolean
        True if the array is a structured array, False otherwise.
    """
    if not isinstance(array, np.ndarray):
        raise TypeError('The input should be a numpy array-like.')

    return len(array.dtype) != 0


def indices_by_type(array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identifies indices of columns with numerical and non-numerical values.

    Checks whether a numpy array is purely numerical or a structured array
    and returns two numpy arrays: the first-one with indices of numerical
    columns and the second-one with indices of non-numerical columns.

    Parameters
    ----------
    array : numpy.ndarray
        A numpy array to be checked (it has to be a 2-dimensional array).

    Raises
    ------
    TypeError
        The input array is not a numpy array-like object.
    ValueError
        The input array consists of complex types such as numpy void and
        object-like types that are not supported by this function.
    IncorrectShapeError
        The input array is not 2-dimensional.

    Returns
    -------
    numerical_indices : numpy.ndarray
        A numpy array containing indices of the numerical columns of the input
        array.
    non_numerical_indices : numpy.ndarray
        A numpy array containing indices of the non-numerical columns of the
        input array.
    """
    if not isinstance(array, np.ndarray):
        raise TypeError('The input should be a numpy array-like.')
    if not is_2d_array(array):
        raise IncorrectShapeError('The input array should be 2-dimensional.')
    if not is_base_array(array):
        raise ValueError('indices_by_type only supports input arrays that '
                         'hold base numpy types, i.e. numerical and '
                         'string-like -- numpy void and object-like types are '
                         'not allowed.')

    if is_structured_array(array):
        assert len(array.dtype) > 1, 'This should be a 2D array.'
        numerical_indices_list = []
        non_numerical_indices_list = []

        for column_name in array.dtype.names:
            column_dtype = array.dtype[column_name]
            if is_numerical_dtype(column_dtype):
                numerical_indices_list.append(column_name)
            else:
                non_numerical_indices_list.append(column_name)

        numerical_indices = np.array(numerical_indices_list)
        non_numerical_indices = np.array(non_numerical_indices_list)
    else:
        if is_numerical_array(array):
            numerical_indices = np.array(range(array.shape[1]))
            non_numerical_indices = np.empty((0, ), dtype='i8')
        else:
            numerical_indices = np.empty((0, ), dtype='i8')
            non_numerical_indices = np.array(range(array.shape[1]))

    return numerical_indices, non_numerical_indices


def get_invalid_indices(array: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    Returns a numpy array with column indices that the input array is missing.

    Parameters
    ----------
    array : numpy.ndarray
        A 2-dimensional array to be checked.
    indices : numpy.ndarray
        A 1-dimensional array of indices corresponding to columns in the input
        array.

    Raises
    ------
    TypeError
        Either of the input arrays is not a numpy array-like object.
    IncorrectShapeError
        The input array is not 2-dimensional or the indices arrays in not
        1-dimensional.

    Returns
    -------
    invalid_indices : numpy.ndarray
        A **sorted** array of indices that were not found in the input array.
    """
    if not (isinstance(array, np.ndarray) and isinstance(indices, np.ndarray)):
        raise TypeError('Input arrays should be numpy array-like objects.')
    if not is_2d_array(array):
        raise IncorrectShapeError('The input array should be 2-dimensional.')
    if not is_1d_array(indices):
        raise IncorrectShapeError('The indices array should be 1-dimensional.')

    if is_structured_array(array):
        array_indices = set(array.dtype.names)
    else:
        array_indices = set(range(array.shape[1]))

    # Alternatively use numpy's np.isin (which supersedes np.in1d):
    # invalid_indices = indices[np.isin(indices, array_indices, invert=True)]
    # or np.setdiff1d: invalid_indices = np.setdiff1d(indices, array_indices)
    invalid_indices = set(indices.tolist()) - array_indices
    return np.sort(list(invalid_indices))


def are_indices_valid(array: np.array, indices: np.array) -> bool:
    """
    Checks whether all the input ``indices`` are valid for the input ``array``.

    Parameters
    ----------
    array : numpy.array
        The 2-dimensional array to be checked.
    indices : numpy.array
        1-dimensional array of column indices.

    Raises
    ------
    TypeError
        Either of the input arrays is not a numpy array-like object.
    IncorrectShapeError
        The input array is not 2-dimensional or the indices arrays in not
        1-dimensional.

    Returns
    -------
    is_valid : boolean
        A Boolean variable that indicates whether the input column indices are
        valid indices for the input array.
    """
    if not (isinstance(array, np.ndarray) and isinstance(indices, np.ndarray)):
        raise TypeError('Input arrays should be numpy array-like objects.')
    if not is_2d_array(array):
        raise IncorrectShapeError('The input array should be 2-dimensional.')
    if not is_1d_array(indices):
        raise IncorrectShapeError('The indices array should be 1-dimensional.')

    invalid_indices = get_invalid_indices(array, indices)
    assert is_1d_array(invalid_indices), 'This should be a 1-d array.'

    is_valid = not bool(invalid_indices.shape[0])
    return is_valid


def check_model_functionality(model_object: object,
                              require_probabilities: bool = False,
                              suppress_warning: bool = False) -> bool:
    """
    Checks whether a model object has all the required functionality.

    Examines a ``model_object`` and ensures that it has all the required
    methods with the correct number of parameters (excluding ``self``):
    ``__init__`` (at least 0), ``fit`` (at least 2), ``predict`` (at least 1)
    and, if required (``require_probabilities=True``), ``predict_proba`` (at
    least 1).

    Parameters
    ----------
    model_object : object
        A Python object that represents a predictive model.
    require_probabilities : boolean, optional (default=False)
        A boolean parameter that indicates whether the model object should
        contain a ``predict_proba`` method. Defaults to False.
    suppress_warning : boolean, optional (default=False)
        A boolean parameter that indicates whether the function should suppress
        its warning message. Defaults to False.

    Warns
    -----
    UserWarning
        Warns about the required functionality that the model object lacks.

    Returns
    -------
    is_functional : boolean
        A Boolean variable that indicates whether the model object has all the
        desired functionality.
    """
    is_functional = True

    methods = {'fit': 2, 'predict': 1}
    if require_probabilities:
        methods['predict_proba'] = 1

    message_strings = []
    for method in methods:
        if not hasattr(model_object, method):
            is_functional = False
            message_strings.append(
                'The model class is missing \'{}\' method.'.format(method))
        else:
            method_object = getattr(model_object, method)
            required_param_n = 0
            params = inspect.signature(method_object).parameters
            for param in params:
                if params[param].default is params[param].empty:
                    required_param_n += 1
            if required_param_n != methods[method]:
                is_functional = False
                message_strings.append(
                    ('The \'{}\' method of the class has incorrect number '
                     '({}) of the required parameters. It needs to have '
                     'exactly {} required parameters. Try using optional '
                     'parameters if you require more functionality.').format(
                         method, required_param_n, methods[method]))

    if not is_functional and not suppress_warning:
        message = '\n'.join(message_strings)
        warnings.warn(message, category=UserWarning)

    return is_functional
