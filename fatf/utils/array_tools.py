"""
Holds numpy array tools.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import logging
import numpy as np
import numpy.lib.recfunctions as recfn

from typing import List, Optional, Union

import fatf.utils.tools as fut
import fatf.utils.validation as fuv

# This variable decides whether to use local or numpy's built-in
# structured_to_unstructured function. It is assigned by calling
# _choose_structured_to_unstructured_version function at the end of this module
# file -- for details see the docstring of this function.
_LOCAL_STRUCTURED_TO_UNSTRUCTURED = None

__all__ = ['structured_to_unstructured_row',
           'structured_to_unstructured',
           'as_unstructured']

logger = logging.getLogger(__name__)


def _choose_structured_to_unstructured() -> bool:
    """
    Decides which implementation of ``structured_to_unstructured`` to use.

    See :func:`fatf.utils.tools.structured_to_unstructured` function
    description for details on how the choice is made.

    Returns
    -------
    use_local_implementation : boolean
        ``True`` if local implementation
        (:func:`fatf.utils.tools.fatf_structured_to_unstructured) is to be
        used. ``False`` if numpy's implementation
        (:func:`numpy.lib.recfunctions.structured_to_unstructured`) is to be
        used.
    """
    use_local_implementation = True
    np_ver = [int(i) for i in np.version.version.split('.')]
    # Use builtin numpy if it is implemented therein
    if fut.at_least_verion([1, 16], np_ver):
        logger.info("Using numpy's numpy.lib.recfunctions."
                    'structured_to_unstructured as fatf.utils.array_tools.'
                    'structured_to_unstructured and fatf.utils.array_tools.'
                    'structured_to_unstructured_row.')
        use_local_implementation = False
    else:
        logger.info("Using fatf's fatf.utils.array_tools."
                    'fatf_structured_to_unstructured as fatf.utils.'
                    'array_tools.structured_to_unstructured and fatf.utils.'
                    'array_tools.fatf_structured_to_unstructured_row as '
                    'fatf.utils.array_tools.structured_to_unstructured_row.')
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
    if not fuv.is_structured_row(structured_row):
        raise TypeError('The input should be a row of a structured array.')
    for dname in structured_row.dtype.names:
        if not fuv.is_base_dtype(structured_row.dtype[dname]):
            raise ValueError('structured_to_unstructured_row only supports '
                             'conversion of structured rows that hold base '
                             'numpy types, i.e. numerical and string-like -- '
                             'numpy void and object-like types are not '
                             'allowed.')

    assert len(structured_row.dtype.names), 'Structured means non-zero dtype.'
    classic_array = np.array([i for i in structured_row])
    if classic_array.shape[0] == 1:
        unstructured_row = classic_array[0]
    else:
        unstructured_row = classic_array
    return unstructured_row


def structured_to_unstructured_row(
        structured_row: np.void,
        **kwargs: Optional[np.dtype]) -> Union[np.dtype,
                                               np.ndarray]:  # pragma: no cover
    """
    Calls either local or numpy's structured_to_unstructured(_row) function.

    Converts a structured row into an unstructured one using either local
    implementation
    (:func:`fatf.utils.array_tools.fatf_structured_to_unstructured_row`) or
    numpy's own :func:`numpy.lib.recfunctions.structured_to_unstructured`.
    Please see the description of
    :func:`fatf.utils.array_tools.structured_to_unstructured` function for the
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
        (:func:`fatf.utils.array_tools.fatf_structured_to_unstructured_row`).

    Returns
    -------
    classic_row : Union[numpy.dtype, numpy.ndarray]
        A classic numpy array or numpy dtype (in case the structured row has
        just one element) representation of the ``structured_row`` with the
        most generic type out of the input row's dtypes.
    """
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
    if not fuv.is_structured_array(structured_array):
        raise TypeError('structured_array should be a structured numpy array.')
    if not fuv.is_base_array(structured_array):
        raise ValueError('fatf_structured_to_unstructured only supports '
                         'conversion of arrays that hold base numpy types, '
                         'i.e. numerical and string-like -- numpy void and '
                         'object-like types are not allowed.')

    if fuv.is_numerical_array(structured_array):
        dtype = np.array([i for i in structured_array[0]]).dtype
    else:
        dtype = str
    dtyped_columns = []
    assert len(structured_array.dtype.names) != 0, 'This should be structured.'
    for i in structured_array.dtype.names:
        dtyped_columns.append(structured_array[i].astype(dtype))
    classic_array = np.column_stack(dtyped_columns)
    return classic_array


def structured_to_unstructured(
        structured_array: np.ndarray,
        **kwargs: Optional[np.dtype]) -> np.ndarray:  # pragma: no cover
    """
    Calls either local or numpy's structured_to_unstructured function.

    numpy 1.16.0 has introduced
    :func:`numpy.lib.recfunctions.structured_to_unstructured` function. To
    ensure backwards compatibility up to numpy 1.8.2 this package implements
    its own version of this function
    (:func:`fatf.utils.array_tools.fatf_structured_to_unstructured`).
    This function calls the latter if numpy version below 1.16.0 is installed.
    However, if numpy 1.16.0 or above is detected, numpy's implementation is
    used instead.

    For the description of ``structured_to_unstructured`` functionality either
    refer to the corresponding numpy
    (:func:`numpy.lib.recfunctions.structured_to_unstructured`) or local
    (:func:`fatf.utils.array_tools.fatf_structured_to_unstructured`)
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
        (:func:`fatf.utils.array_tools.fatf_structured_to_unstructured`).

    Returns
    -------
    classic_array : numpy.ndarray
        A classic numpy array representation of the ``structured_array`` with
        the most generic type out of the input array's dtypes.
    """
    if _LOCAL_STRUCTURED_TO_UNSTRUCTURED:
        classic_array = fatf_structured_to_unstructured(structured_array)
    else:
        classic_array = recfn.structured_to_unstructured(structured_array,
                                                         **kwargs)
        if (fuv.is_2d_array(structured_array)
                and fuv.is_1d_array(classic_array)):
            classic_array = classic_array.reshape(
                (structured_array.shape[0], 1))
    return classic_array


def as_unstructured(array_like: Union[np.ndarray, np.void],
                    **kwargs: Optional[np.dtype]) -> Union[np.dtype,
                                                           np.ndarray]:
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
        :func:`fatf.utils.array_tools.structured_to_unstructured_row` or
        :func:`fatf.utils.array_tools.structured_to_unstructured` documentation
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
        assert fuv.is_structured_row(array_like), \
            'numpy.void has to be a row of a structured numpy array.'
        classic_array = structured_to_unstructured_row(array_like, **kwargs)
    elif isinstance(array_like, np.ndarray):
        if fuv.is_structured_array(array_like):
            classic_array = structured_to_unstructured(array_like, **kwargs)
        else:
            if fuv.is_base_array(array_like):
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
