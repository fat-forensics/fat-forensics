"""
The :mod:`fatf.transparency.data.describe_functions` module implements
functions to describe data sets.
"""
# Author: Rafael Poyiadzi <rp13102@bristol.ac.uk>
#         Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import warnings

from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np

import fatf.utils.array.tools as fuat
import fatf.utils.array.validation as fuav

from fatf.exceptions import IncorrectShapeError

__all__ = ['describe_array',
           'describe_numerical_array',
           'describe_categorical_array']  # yapf: disable

IndicesType = Set[Union[str, int]]

NUMERICAL_KEYS = [
    'count', 'mean', 'std', 'max', 'min', '25%', '50%', '75%', 'nan_count'
]
CATEGORICAL_KEYS = [
    'count', 'unique', 'unique_counts', 'top', 'freq', 'is_top_unique'
]


def describe_array(
        array: np.ndarray,
        include: Optional[Union[str, int, List[Union[str, int]]]] = None,
        exclude: Optional[Union[str, int, List[Union[str, int]]]] = None,
        **kwargs: bool
) -> Dict[Union[str, int],
          Union[str, int, float, bool, np.ndarray,
                Dict[str, Union[str, int, float, bool, np.ndarray]]]
          ]:  # yapf: disable
    """
    Describes categorical (textual) and numerical columns in the input array.

    The details of numerical and categorical descriptions can be found in
    :func:`fatf.transparency.data.describe_functions.describe_numerical_array`
    and :func:`fatf.transparency.data.describe_functions.\
describe_categorical_array` functions documentation respectively.

    To filter out the columns that will be described you can use ``include``
    and ``exclude`` parameters. Either of these can be a list with columns
    indices, a string or an integer when excluding or including just one
    column; or one of the keywords: ``'numerical'`` or ``'categorical'``, to
    indicate that only numerical or categorical columns should be included/
    excluded. By default all columns are described.

    Parameters
    ----------
    array : numpy.ndarray
        The array to be described.
    include : Union[str, int, List[Union[str, int]]], optional (default=None)
        A list of column indices to be included in the description. If
        ``None`` (the default value), all of the columns will be included.
        Alternatively this can be set to a single index (either a string or an
        integer) to compute statistics just for this one column. It is also
        possible to set it to ``'numerical'`` or ``'categorical'`` to just
        include numerical or categorical columns respectively.
    exclude : Union[str, int, List[Union[str, int]]], optional (default=None)
        A list of column indices to be excluded from the description. If
        ``None`` (the default value), none of the columns will be excluded.
        Alternatively this can be set to a single index (either a string or an
        integer) to exclude just one column. It is also possible to set it to
        ``'numerical'`` or ``'categorical'`` to exclude wither all numerical or
        all categorical columns respectively.
    **kwargs : bool
        Keyword arguments that are passed to the :func:`fatf.transparency.\
data.describe_functions.describe_numerical_array` function responsible for
        describing numerical arrays.

    Warns
    -----
    UserWarning
        When using ``include`` or ``exclude`` parameters for 1-dimensional
        input arrays (in which case these parameters are ignored).

    Raises
    ------
    IncorrectShapeError
        The input array is neither 1- not 2-dimensional.
    RuntimeError
        None of the columns were selected to be described.
    ValueError
        The input array is not of a base type (textual and numerical elements).
        The input array has 0 columns.

    Returns
    -------
    description : Dict[Union[str, int], Dict[str, \
Union[str, int, float bool, np.ndarray]]]
        For 2-dimensional arrays a dictionary describing every column under a
        key corresponding to its index in the input array. For a 1-dimensional
        input array a dictionary describing that array.
    """
    # pylint: disable=too-many-locals,too-many-branches
    is_1d = fuav.is_1d_like(array)
    if is_1d:
        array = fuat.as_unstructured(array)
        is_2d = False
    else:
        is_2d = fuav.is_2d_array(array)

    if not is_1d and not is_2d:
        raise IncorrectShapeError('The input array should be 1- or '
                                  '2-dimensional.')

    if not fuav.is_base_array(array):
        raise ValueError('The input array should be of a base type (a mixture '
                         'of numerical and textual types).')

    if is_1d:
        if include is not None or exclude is not None:
            warnings.warn(
                'The input array is 1-dimensional. Ignoring include and '
                'exclude parameters.',
                category=UserWarning)

        if fuav.is_numerical_array(array):
            description = describe_numerical_array(array, **kwargs)
        elif fuav.is_textual_array(array):
            description = describe_categorical_array(array)
        else:  # pragma: no cover
            assert False, 'A base array should either be numerical or textual.'
    elif is_2d:
        numerical_indices, categorical_indices = fuat.indices_by_type(array)
        is_structured_array = fuav.is_structured_array(array)

        if (numerical_indices.shape[0] + categorical_indices.shape[0]) == 0:
            raise ValueError('The input array cannot have 0 columns.')

        numerical_indices_set = set(numerical_indices)
        categorical_indices_set = set(categorical_indices)
        all_indices = categorical_indices_set.union(numerical_indices_set)
        # Indices to be included
        include_indices = _filter_include_indices(categorical_indices_set,
                                                  numerical_indices_set,
                                                  include, all_indices)
        categorical_indices_set, numerical_indices_set = include_indices

        # Indices to be included
        exclude_indices = _filter_exclude_indices(categorical_indices_set,
                                                  numerical_indices_set,
                                                  exclude, all_indices)
        categorical_indices_set, numerical_indices_set = exclude_indices

        all_indices = numerical_indices_set.union(categorical_indices_set)
        if len(all_indices) == 0:  # pylint: disable=len-as-condition
            raise RuntimeError('None of the columns were selected to be '
                               'described.')

        description = dict()
        for idx in numerical_indices_set:
            if is_structured_array:
                description[idx] = describe_numerical_array(  # type: ignore
                    array[idx], **kwargs)
            else:
                description[idx] = describe_numerical_array(  # type: ignore
                    array[:, idx], **kwargs)
        for idx in categorical_indices_set:
            if is_structured_array:
                description[idx] = describe_categorical_array(  # type: ignore
                    array[idx])
            else:
                description[idx] = describe_categorical_array(  # type: ignore
                    array[:, idx])
    else:  # pragma: no cover
        assert False, 'The input array can only be 1- or 2-dimensional.'

    return description  # type: ignore


def describe_numerical_array(array: Union[np.ndarray, np.void],
                             skip_nans: bool = True
                             ) -> Dict[str, Union[int, float, np.ndarray]]:
    """
    Describes a numerical numpy array with basic statistics.

    If the ``skip_nans`` parameter is set to ``True``, any ``numpy.nan``
    present in the input array is skipped for calculating the statistics.
    Otherwise, they are included, affecting most of the statistics and possibly
    equating them to ``numpy.nan``.

    The description output by this function is a dictionary with the
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
        An array for which a description is desired.
    skip_nans : boolean, optional (default=True)
        If set to ``True``, ``numpy.nan``\\ s present in the input array will
        be excluded while computing the statistics.

    Raises
    ------
    IncorrectShapeError
        The input array is not 1-dimensional.
    ValueError
        The input array is not purely numerical or it is empty.

    Returns
    -------
    numerical_description : Dict[string, Union[integer, float, numpy.ndarray]]
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
        '25%': np.percentile(classic_array, 25),
        '50%': np.percentile(classic_array, 50),
        '75%': np.percentile(classic_array, 75),
        'max': np.max(classic_array),
        'nan_count': nan_indices.sum()
    }

    return numerical_description


def describe_categorical_array(
        array: Union[np.ndarray, np.void]
) -> Dict[str, Union[str, int, bool, np.ndarray]]:
    """
    Describes a categorical numpy array with basic statistics.

    The description output by this function is a dictionary with the
    following keys:

    ``count`` : integer
        The number of elements in the array.

    ``unique`` : numpy.ndarray
        The unique values in the array, ordered lexicographically.

    ``unique_counts`` : numpy.ndarray
        The counts of the unique values in the array.

    ``top`` : string
        The most frequent value in the array.

    ``freq`` : integer
        The count of the most frequent value in the array.

    ``is_top_unique`` : boolean
        Indicates whether the most frequent value (``freq``) in the array is
        the only one with that count.

    Parameters
    ----------
    array : Union[numpy.ndarray, numpy.void]
        An array for which a description is desired.

    Raises
    ------
    IncorrectShapeError
        The input array is not 1-dimensinoal.
    ValueError
        The input array is empty.

    Warns
    -----
    UserWarning
        When the input array is not purely textual it needs to be converted to
        a string type before it can be described.

    Returns
    -------
    categorical_description : Dict[string, Union[string, integer, \
boolean, numpy.ndarray]]
        A dictionary describing the categorical input array.
    """
    if not fuav.is_1d_like(array):
        raise IncorrectShapeError('The input array should be 1-dimensional.')

    classic_array = fuat.as_unstructured(array)
    assert len(classic_array.shape) == 1, '1D arrays only at this point.'

    if not classic_array.shape[0]:
        raise ValueError('The input array cannot be empty.')
    if not fuav.is_textual_array(classic_array):
        warnings.warn(
            'The input array is not purely categorical. Converting the input '
            'array into a textual type to facilitate a categorical '
            'description.',
            category=UserWarning)
        classic_array = classic_array.astype(str)

    unique, unique_counts = np.unique(classic_array, return_counts=True)

    unique_sort_index = np.argsort(unique)
    unique = unique[unique_sort_index]
    unique_counts = unique_counts[unique_sort_index]

    top_index = np.argmax(unique_counts)

    top = unique[top_index]
    freq = unique_counts[top_index]

    is_top_unique = (unique_counts == freq).sum() < 2

    categorical_description = {
        'count': classic_array.shape[0],
        'unique': unique,
        'unique_counts': unique_counts,
        'top': top,
        'freq': freq,
        'is_top_unique': is_top_unique
    }

    return categorical_description


def _filter_include_indices(
        categorical_indices_set: IndicesType,
        numerical_indices_set: IndicesType,
        include: Union[None, str, int, List[Union[str, int]]],
        all_indices: IndicesType) -> Tuple[IndicesType, IndicesType]:
    """
    Filters categorical and numerical indices sets with the include set.

    For a detailed description of the filtering mechanism please refer to
    :func:`fatf.transparency.data.describe_functions.describe_array`
    documentation.

    Parameters
    ----------
    categorical_indices_set : Set[Union[string, integer]]
        A set of categorical indices to be filtered.
    numerical_indices_set : Set[Union[string, integer]]
        A set of numerical indices to be filtered.
    include : Union[None, string, integer, List[Union[string, integer]]]
        An index or a list of indices to be included. ``None`` means including
        all of the indices.
    all_indices : Set[Union[string, integer]]
        A set of all indices before any filtering was ever applied.

    Raises
    ------
    TypeError
        The ``include`` parameter is of a wrong type.
    IndexError
        Indices given in the ``include`` parameter are not valid indices for
        the ``categorical_indices_set`` and ``numerical_indices_set``.

    Returns
    -------
    categorical_indices_set : Set[Union[string, integer]]
        A set of categorical indices after filtering.
    numerical_indices_set : Set[Union[string, integer]]
        A set of numerical indices after filtering.
    """
    assert isinstance(categorical_indices_set, set), 'This has to be a set.'
    assert isinstance(numerical_indices_set, set), 'This has to be a set.'

    if include is None:
        pass
    elif isinstance(include, (str, int)):
        if include == 'numerical':
            categorical_indices_set = set()
        elif include == 'categorical':
            numerical_indices_set = set()
        else:
            if include not in all_indices:
                raise IndexError('The following include index is not a valid '
                                 'index: {}.'.format(include))

            numerical_indices_set = numerical_indices_set.intersection(
                [include])
            categorical_indices_set = categorical_indices_set.intersection(
                [include])
    elif isinstance(include, list):
        invalid_indices = set(include).difference(all_indices)
        if invalid_indices:
            raise IndexError('The following include indices are not valid '
                             'indices: {}.'.format(invalid_indices))

        numerical_indices_set = numerical_indices_set.intersection(include)
        categorical_indices_set = categorical_indices_set.intersection(include)
    else:
        raise TypeError('The include parameter can either be a string, an '
                        'integer or a list of these two types.')

    return categorical_indices_set, numerical_indices_set


def _filter_exclude_indices(
        categorical_indices_set: IndicesType,
        numerical_indices_set: IndicesType,
        exclude: Union[None, str, int, List[Union[str, int]]],
        all_indices: IndicesType) -> Tuple[IndicesType, IndicesType]:
    """
    Filters categorical and numerical indices sets with the exclude set.

    For a detailed description of the filtering mechanism please refer to
    :func:`fatf.transparency.data.describe_functions.describe_array`
    documentation.

    Parameters
    ----------
    categorical_indices_set : Set[Union[string, integer]]
        A set of categorical indices to be filtered.
    numerical_indices_set : Set[Union[string, integer]]
        A set of numerical indices to be filtered.
    exclude : Union[None, string, integer, List[Union[string, integer]]]
        An index or a list of indices to be excluded. ``None`` means not
        excluding any index.
    all_indices : Set[Union[string, integer]]
        A set of all indices before any filtering was ever applied.

    Raises
    ------
    TypeError
        The ``exclude`` parameter is of a wrong type.
    IndexError
        Indices given in the ``exclude`` parameter are not valid indices for
        the ``categorical_indices_set`` and ``numerical_indices_set``.

    Returns
    -------
    categorical_indices_set : Set[Union[string, integer]]
        A set of categorical indices after filtering.
    numerical_indices_set : Set[Union[string, integer]]
        A set of numerical indices after filtering.
    """
    assert isinstance(categorical_indices_set, set), 'This has to be a set.'
    assert isinstance(numerical_indices_set, set), 'This has to be a set.'

    if exclude is None:
        pass
    elif isinstance(exclude, (str, int)):
        if exclude == 'numerical':
            numerical_indices_set = set()
        elif exclude == 'categorical':
            categorical_indices_set = set()
        else:
            if exclude not in all_indices:
                raise IndexError('The following exclude index is not a valid '
                                 'index: {}.'.format(exclude))

            numerical_indices_set = numerical_indices_set.difference([exclude])
            categorical_indices_set = categorical_indices_set.difference(
                [exclude])
    elif isinstance(exclude, list):
        invalid_indices = set(exclude).difference(all_indices)
        if invalid_indices:
            raise IndexError('The following exclude indices are not valid '
                             'indices: {}.'.format(invalid_indices))

        numerical_indices_set = numerical_indices_set.difference(exclude)
        categorical_indices_set = categorical_indices_set.difference(exclude)
    else:
        raise TypeError('The exclude parameter can either be a string, an '
                        'integer or a list of these two types.')

    return categorical_indices_set, numerical_indices_set
