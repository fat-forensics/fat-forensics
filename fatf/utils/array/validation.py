"""
The :mod:`fatf.utils.array.validation` module holds numpy array validators.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import warnings

from typing import Union

import numpy as np

import fatf.utils.tools as fut

__all__ = ['is_numerical_dtype',
           'is_textual_dtype',
           'is_base_dtype',
           'is_flat_dtype',
           'are_similar_dtypes',
           'are_similar_dtype_arrays',
           'is_numerical_array',
           'is_textual_array',
           'is_base_array',
           'is_1d_array',
           'is_2d_array',
           'is_structured_row',
           'is_1d_like',
           'is_structured_array']  # yapf: disable

_NUMPY_VERSION = [int(i) for i in np.version.version.split('.')]
_NUMPY_1_13 = fut.at_least_verion([1, 13], _NUMPY_VERSION)

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
        warnings.warn(
            'Zero-terminated bytes type is not supported and is not '
            'considered to be a textual type. Please use any other textual '
            'type.',
            category=UserWarning)
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


def is_flat_dtype(dtype: np.dtype) -> bool:
    """
    Determines whether a numpy dtype object is flat.

    Checks whether the ``dtype`` just encodes one element or a shape. A dtype
    can characterise an array of other base types, which can then be embedded
    as an element of another array.

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
    is_flat : boolean
        True if the dtype is flat, False otherwise.
    """
    if not isinstance(dtype, np.dtype):
        raise TypeError('The input should be a numpy dtype object.')

    # If the dtype is complex
    if dtype.names is not None:
        raise ValueError('The numpy dtype object is structured. '
                         'Only base dtype are allowed.')

    # pylint: disable=len-as-condition
    if _NUMPY_1_13:  # pragma: no cover
        is_flat = not bool(dtype.ndim)
    else:  # pragma: no cover
        is_flat = len(dtype.shape) == 0

    return is_flat


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
        The second dtype to be compared.
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
        are_similar = are_similar_dtypes(array_a.dtype, array_b.dtype,
                                         strict_comparison)
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
        # pylint: disable=len-as-condition
        if len(array.shape) == 2 and len(array.dtype) == 1:
            is_2d = False
            message = ('2-dimensional arrays with 1D structured elements are '
                       'not acceptable. Such a numpy array can be expressed '
                       'as a classic 2D numpy array with a desired type.')
            warnings.warn(message, category=UserWarning)
        elif len(array.shape) == 1 and len(array.dtype) > 0:
            is_2d = True
            for name in array.dtype.names:
                if not is_flat_dtype(array.dtype[name]):
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
    is_1d_like_array : boolean
        True if the input is either a 1-dimensional numpy array or a row of a
        structured numpy array, False otherwise.
    """
    is_1d_like_array = False
    if isinstance(oned_like_object, np.void):
        is_1d_like_array = is_structured_row(oned_like_object)
    elif isinstance(oned_like_object, np.ndarray):
        is_1d_like_array = is_1d_array(oned_like_object)
    else:
        raise TypeError('The input should either be a numpy array-like object '
                        '(numpy.ndarray) or a row of a structured numpy array '
                        '(numpy.void).')

    return is_1d_like_array


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
