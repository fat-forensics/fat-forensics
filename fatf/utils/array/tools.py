"""
The :mod:`fatf.utils.array.tools` module implements tools for numpy array.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import logging

from typing import Optional, Tuple, Union

import numpy as np
import numpy.lib.recfunctions as recfn

import fatf.utils.tools as fut
import fatf.utils.array.validation as fuav

from fatf.exceptions import IncorrectShapeError

# This variable decides whether to use local or numpy's built-in
# structured_to_unstructured function. It is assigned by calling
# _choose_structured_to_unstructured_version function at the end of this module
# file -- for details see the docstring of this function.
_LOCAL_STRUCTURED_TO_UNSTRUCTURED = None

__all__ = ['indices_by_type',
           'get_invalid_indices',
           'are_indices_valid',
           'generalise_dtype',
           'structured_to_unstructured_row',
           'structured_to_unstructured',
           'as_unstructured']  # yapf: disable

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


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
    if not fuav.is_2d_array(array):
        raise IncorrectShapeError('The input array should be 2-dimensional.')
    if not fuav.is_base_array(array):
        raise ValueError('indices_by_type only supports input arrays that '
                         'hold base numpy types, i.e. numerical and '
                         'string-like -- numpy void and object-like types are '
                         'not allowed.')

    if fuav.is_structured_array(array):
        assert len(array.dtype) > 1, 'This should be a 2D array.'
        numerical_indices_list = []
        non_numerical_indices_list = []

        for column_name in array.dtype.names:
            column_dtype = array.dtype[column_name]
            if fuav.is_numerical_dtype(column_dtype):
                numerical_indices_list.append(column_name)
            else:
                non_numerical_indices_list.append(column_name)

        numerical_indices = np.array(numerical_indices_list)
        non_numerical_indices = np.array(non_numerical_indices_list)
    else:
        if fuav.is_numerical_array(array):
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
    if not fuav.is_2d_array(array):
        raise IncorrectShapeError('The input array should be 2-dimensional.')
    if not fuav.is_1d_array(indices):
        raise IncorrectShapeError('The indices array should be 1-dimensional.')

    if fuav.is_structured_array(array):
        array_indices = set(array.dtype.names)
    else:
        array_indices = set(range(array.shape[1]))

    # Alternatively use numpy's np.isin (which supersedes np.in1d):
    # invalid_indices = indices[np.isin(indices, array_indices, invert=True)]
    # or np.setdiff1d: invalid_indices = np.setdiff1d(indices, array_indices)
    invalid_indices = set(indices.tolist()) - array_indices
    return np.sort(list(invalid_indices))


def are_indices_valid(array: np.ndarray, indices: np.ndarray) -> bool:
    """
    Checks whether all the input ``indices`` are valid for the input ``array``.

    Parameters
    ----------
    array : numpy.ndarray
        The 2-dimensional array to be checked.
    indices : numpy.ndarray
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
    if not fuav.is_2d_array(array):
        raise IncorrectShapeError('The input array should be 2-dimensional.')
    if not fuav.is_1d_array(indices):
        raise IncorrectShapeError('The indices array should be 1-dimensional.')

    invalid_indices = get_invalid_indices(array, indices)
    assert fuav.is_1d_array(invalid_indices), 'This should be a 1-d array.'

    is_valid = not bool(invalid_indices.shape[0])
    return is_valid


def generalise_dtype(dtype_one: np.dtype, dtype_two: np.dtype) -> np.dtype:
    """
    Finds the more general type of the two given.

    Parameters
    ----------
    dtype_one : numpy.dtype
        The first dtype to be compared.
    dtype_two : numpy.dtype
        The second dtype to be compared.

    Raises
    ------
    ValueError
        Either of the input dtypes is not a base dtype: either textual or
        numerical type.

    Returns
    -------
    common_dtype : numpy.dtype
        The more general type of the two given as the input parameters.
    """
    error_msg = 'The {} dtype is not one of the base types (strings/numbers).'
    if not fuav.is_base_dtype(dtype_one):
        raise ValueError(error_msg.format('first'))
    if not fuav.is_base_dtype(dtype_two):
        raise ValueError(error_msg.format('second'))

    type_one_array = np.ones((1, ), dtype=dtype_one)
    type_two_array = np.ones((1, ), dtype=dtype_two)
    common_array = np.concatenate([type_one_array, type_two_array])
    common_dtype = common_array.dtype

    return common_dtype


def _choose_structured_to_unstructured() -> bool:
    """
    Decides which implementation of ``structured_to_unstructured`` to use.

    See :func:`fatf.utils.tools.structured_to_unstructured` function
    description for details on how the choice is made.

    Returns
    -------
    use_local_implementation : boolean
        ``True`` if local implementation
        (``fatf.utils.tools.fatf_structured_to_unstructured``) is to be
        used. ``False`` if numpy's implementation
        (``numpy.lib.recfunctions.structured_to_unstructured``) is to be used.
    """
    use_local_implementation = True
    np_ver = [int(i) for i in np.version.version.split('.')]
    # Use builtin numpy if it is implemented therein
    if fut.at_least_verion([1, 16], np_ver):
        logger.info("Using numpy's numpy.lib.recfunctions."
                    'structured_to_unstructured as fatf.utils.array.tools.'
                    'structured_to_unstructured and fatf.utils.array.tools.'
                    'structured_to_unstructured_row.')
        use_local_implementation = False
    else:
        logger.info("Using fatf's fatf.utils.array.tools."
                    'fatf_structured_to_unstructured as fatf.utils.'
                    'array.tools.structured_to_unstructured and fatf.utils.'
                    'array.tools.fatf_structured_to_unstructured_row as '
                    'fatf.utils.array.tools.structured_to_unstructured_row.')
        use_local_implementation = True
    return use_local_implementation


def fatf_structured_to_unstructured_row(
        structured_row: np.void) -> Union[np.dtype, np.ndarray]:
    """
    Converts a structured row into a 1D plane array of the most generic type.

    If the input row is purely numerical, the output array is of the most
    generic numerical type. Otherwise, the output arrays is converted to a
    string type.

    Parameters
    ----------
    structured_row : numpy.void
        A row extracted from a structured numpy array that will be converted
        into a plane 1-dimensional numpy array.

    Raises
    ------
    TypeError
        The input is not a row of a structured numpy array.
    ValueError
        The input row consists of complex types such as numpy void and
        object-like types that are not supported by this function.

    Returns
    -------
    unstructured_row : Union[numpy.dtype, numpy.ndarray]
        A classic numpy array or numpy dtype (in case the structured row has
        just one element) representation of the ``structured_row`` with the
        most generic type out of the input row's dtypes.
    """
    if not fuav.is_structured_row(structured_row):
        raise TypeError('The input should be a row of a structured array.')
    for dname in structured_row.dtype.names:
        if not fuav.is_base_dtype(structured_row.dtype[dname]):
            raise ValueError('structured_to_unstructured_row only supports '
                             'conversion of structured rows that hold base '
                             'numpy types, i.e. numerical and string-like -- '
                             'numpy void and object-like types are not '
                             'allowed.')

    # pylint: disable=len-as-condition
    assert len(structured_row.dtype.names), 'Structured means non-zero dtype.'
    classic_array = np.array([i for i in structured_row])
    if classic_array.shape[0] == 1:
        unstructured_row = classic_array[0]
    else:
        unstructured_row = classic_array
    return unstructured_row


def structured_to_unstructured_row(
        structured_row: np.void, **kwargs: Optional[np.dtype]
) -> Union[np.dtype, np.ndarray]:  # pragma: no cover
    """
    Calls either local or numpy's ``structured_to_unstructured`` function.

    Converts a structured row into an unstructured one using either local
    implementation
    (``fatf.utils.array.tools.fatf_structured_to_unstructured_row``) or
    numpy's own ``numpy.lib.recfunctions.structured_to_unstructured``.
    Please see the description of
    :func:`fatf.utils.array.tools.structured_to_unstructured` function for the
    detailed description of when a particular implementation is chosen.

    .. warning:: Since this function either calls a local implementation or a
       builtin numpy function there may be some inconsistencies in its
       behaviour. One that we are aware of is conversion of arrays that contain
       ``'V'`` -- raw data (void), ``'O'`` -- (Python) objects, ``'M'`` --
       datetime or ``'m'`` -- timedelta dtypes. These types are not supported
       by the local implementation, however some of them are supported by the
       numpy built-in, e.g. the ``'V'`` type.

    Parameters
    ----------
    structured_row : numpy.void
        A row of a structured numpy array to be converted into a plane numpy
        array representation.
    **kwargs : Optional[numpy.dtype]
        Named parameters that are passed to the appropriate structured to
        unstructured array converter. These parameters are ignored when calling
        the local implementation
        (``fatf.utils.array.tools.fatf_structured_to_unstructured_row``).

    Returns
    -------
    classic_row : Union[numpy.dtype, numpy.ndarray]
        A classic numpy array or numpy dtype (in case the structured row has
        just one element) representation of the ``structured_row`` with the
        most generic type out of the input row's dtypes.
    """
    # pylint: disable=no-member
    if _LOCAL_STRUCTURED_TO_UNSTRUCTURED:
        classic_row = fatf_structured_to_unstructured_row(structured_row)
    else:
        classic_row = recfn.structured_to_unstructured(structured_row,
                                                       **kwargs)
    return classic_row


def fatf_structured_to_unstructured(
        structured_array: np.ndarray) -> np.ndarray:
    """
    Converts a structured array into a plane array of the most generic type.

    If the input arrays is purely numerical, the output array is of the most
    generic numerical type. Otherwise, the output arrays is converted to a
    string type.

    Parameters
    ----------
    structured_array : numpy.ndarray
        A structured numpy array to be converted into a plane numpy array.

    Raises
    ------
    TypeError
        The input array is not a structured numpy array.
    ValueError
        The input array consists of complex types such as numpy void and
        object-like types that are not supported by this function.

    Returns
    -------
    classic_array : numpy.ndarray
        A classic numpy array representation of the ``structured_array`` with
        the most generic type out of the input array's dtypes.
    """
    if not fuav.is_structured_array(structured_array):
        raise TypeError('structured_array should be a structured numpy array.')
    if not fuav.is_base_array(structured_array):
        raise ValueError('fatf_structured_to_unstructured only supports '
                         'conversion of arrays that hold base numpy types, '
                         'i.e. numerical and string-like -- numpy void and '
                         'object-like types are not allowed.')

    if fuav.is_numerical_array(structured_array):
        dtype = np.array([i for i in structured_array[0]]).dtype
    else:
        dtype = str
    dtyped_columns = []
    # pylint: disable=len-as-condition
    assert len(structured_array.dtype.names) != 0, 'This should be structured.'
    for i in structured_array.dtype.names:
        dtyped_columns.append(structured_array[i].astype(dtype))
    classic_array = np.column_stack(dtyped_columns)
    return classic_array


def structured_to_unstructured(
        structured_array: np.ndarray,
        **kwargs: Optional[np.dtype]) -> np.ndarray:  # pragma: no cover
    """
    Calls either local or numpy's ``structured_to_unstructured`` function.

    numpy 1.16.0 has introduced
    ``numpy.lib.recfunctions.structured_to_unstructured`` function. To
    ensure backwards compatibility up to numpy 1.9.0 this package implements
    its own version of this function
    (``fatf.utils.array.tools.fatf_structured_to_unstructured``).
    This function calls the latter if numpy version below 1.16.0 is installed.
    However, if numpy 1.16.0 or above is detected, numpy's implementation is
    used instead.

    For the description of ``structured_to_unstructured`` functionality either
    refer to the corresponding numpy
    (``numpy.lib.recfunctions.structured_to_unstructured``) or local
    (``fatf.utils.array.tools.fatf_structured_to_unstructured``)
    documentation.

    .. warning:: Since this function either calls a local implementation or a
       builtin numpy function there may be some inconsistencies in its
       behaviour. One that we are aware of is conversion of arrays that contain
       ``'V'`` -- raw data (void), ``'O'`` -- (Python) objects, ``'M'`` --
       datetime or ``'m'`` -- timedelta dtypes. These types are not supported
       by the local implementation, however some of them are supported by the
       numpy built-in, e.g. the ``'V'`` type.

    Parameters
    ----------
    structured_array : numpy.ndarray
        A structured numpy array to be converted into a plane numpy array.
    **kwargs : Optional[numpy.dtype]
        Named parameters that are passed to the appropriate structured to
        unstructured array converter. These parameters are ignored when calling
        the local implementation
        (``fatf.utils.array.tools.fatf_structured_to_unstructured``).

    Returns
    -------
    classic_array : numpy.ndarray
        A classic numpy array representation of the ``structured_array`` with
        the most generic type out of the input array's dtypes.
    """
    # pylint: disable=no-member
    if _LOCAL_STRUCTURED_TO_UNSTRUCTURED:
        classic_array = fatf_structured_to_unstructured(structured_array)
    else:
        classic_array = recfn.structured_to_unstructured(
            structured_array, **kwargs)
        if (fuav.is_2d_array(structured_array)
                and fuav.is_1d_array(classic_array)):
            classic_array = classic_array.reshape((structured_array.shape[0],
                                                   1))
    return classic_array


def as_unstructured(
        array_like: Union[np.ndarray, np.void],
        **kwargs: Optional[np.dtype]) -> Union[np.dtype, np.ndarray]:
    """
    Converts an array like object into an unstructured array.

    If the input array is unstructured, it is return without any
    transformations. Otherwise, if the input array is either a structured array
    or a structured array row, appropriate structured to unstructured function
    is called.

    .. warning:: Since this function either calls a local implementation or a
       builtin numpy function there may be some inconsistencies in its
       behaviour. One that we are aware of is conversion of arrays that contain
       ``'V'`` -- raw data (void), ``'O'`` -- (Python) objects, ``'M'`` --
       datetime or ``'m'`` -- timedelta dtypes. These types are not supported
       by the local implementation, however some of them are supported by the
       numpy built-in, e.g. the ``'V'`` type.

    Parameters
    ----------
    array_like : Union[numpy.ndarray, numpy.void]
        An array, a structured array or a row of a structured numpy array to be
        converted into a plane numpy array representation.
    **kwargs : Optional[numpy.dtype]
        Named parameters that are passed to the appropriate structured to
        unstructured array converter. These parameters are ignored when calling
        any of the local implementations -- see either
        :func:`fatf.utils.array.tools.structured_to_unstructured_row` or
        :func:`fatf.utils.array.tools.structured_to_unstructured` documentation
        for details.

    Raises
    ------
    TypeError
        The input array is not a numpy array, a structured numpy array or a row
        of a structured numpy array.
    ValueError
        The input array consists of complex types such as numpy void and
        object-like types that are not supported by this function.

    Returns
    -------
    classic_array : Union[numpy.dtype, numpy.ndarray]
        A classic numpy array or numpy dtype (in case the structured row has
        just one element) representation of the ``structured_row`` with the
        most generic type out of the input row's dtypes.
    """
    if isinstance(array_like, np.void):
        assert fuav.is_structured_row(array_like), \
            'numpy.void has to be a row of a structured numpy array.'
        classic_array = structured_to_unstructured_row(array_like, **kwargs)
    elif isinstance(array_like, np.ndarray):
        if fuav.is_structured_array(array_like):
            classic_array = structured_to_unstructured(array_like, **kwargs)
        else:
            if fuav.is_base_array(array_like):
                classic_array = array_like
            else:
                raise ValueError('as_unstructured only supports conversion of '
                                 'arrays that hold base numpy types, i.e. '
                                 'numerical and string-like -- numpy void and '
                                 'object-like types are not allowed.')
    else:
        raise TypeError('The input should either be a numpy (structured or '
                        'unstructured) array-like object (numpy.ndarray) or a '
                        'row of a structured numpy array (numpy.void).')
    return classic_array


# Set the right boolean value to the structured_to_unstructured version chooser
_LOCAL_STRUCTURED_TO_UNSTRUCTURED = _choose_structured_to_unstructured()
assert isinstance(_LOCAL_STRUCTURED_TO_UNSTRUCTURED, bool), \
    'structured_to_unstructured version specifier should be a boolean.'
