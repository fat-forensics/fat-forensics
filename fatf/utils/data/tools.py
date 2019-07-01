"""
The :mod:`fatf.utils.data.tools` module implements tools for data set handling.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
#         Matt Clifford <mc15445@bristol.ac.uk>
# License: new BSD

import inspect
import warnings

from numbers import Number
from typing import Callable, List, Optional, Tuple, Union
from typing import Set  # pylint: disable=unused-import

import numpy as np

import fatf.utils.array.tools as fuat
import fatf.utils.array.validation as fuav

from fatf.exceptions import IncorrectShapeError

__all__ = ['group_by_column', 'apply_to_column_grouping']

Index = Union[int, str]  # A column index type


def group_by_column(dataset: np.ndarray,
                    column_index: Index,
                    groupings: Optional[List[Union[float, Tuple[str]]]] = None,
                    numerical_bins_number: int = 5,
                    treat_as_categorical: Optional[bool] = None
                    ) -> Tuple[List[List[int]], List[str]]:
    """
    Groups row indices of an array based on value grouping of a chosen column.

    If selected column is numerical, by default the values are grouped into 5
    bins equally distributed between the minimum and the maximum value of the
    column. The number of bins can be changed with the
    ``numerical_bins_number`` if desired. Alternatively, the exact bin
    boundaries can be given via the ``groupings`` parameter.

    For categorical columns, the default binning is one bin for every unique
    value in the selected column. This behaviour can be changed by providing
    the ``groupings`` parameter, where multiple values can be selected to
    create one bin.

    Parameters
    ----------
    dataset : numpy.ndarray
        A dataset to be used for grouping the row indices.
    column_index : Union[string, integer]
        A column index (a string for structured numpy arrays or an integer for
        unstructured arrays) of the column based on which the row indices will
        be partitioned.
    groupings : List[Union[number, Tuple[string]]], optional (default=None)
        A list of user-specified groupings for the selected column. The default
        grouping for categorical (textual) columns is splitting them by all the
        unique values therein. The numerical columns are, by default, binned
        into 5 bins (see the ``numerical_bins_number`` parameter) uniformly
        distributed between the minimum and the maximum value of the column.
        To introduce custom binning for a categorical column ``groupings``
        parameter should be a list of tuples, where every tuple represents a
        single group. For example, a column with the following unique values
        ``['a', 'b', 'c', 'd']`` can be split into two groups: ``['a', 'd']``
        and ``['b', 'c']`` by providing ``[('a', 'd'), ('b', 'c')]`` grouping.
        For numerical columns custom grouping should be introduced as a list of
        bucket boundaries. Every bucket includes all the values that are
        **less or equal** to the specified bucket boundary and greater than the
        previous boundary if one is given.
    numerical_bins_number : integer, optional (default=5)
        The number of bins used for default binning of numerical columns.
    treat_as_categorical : boolean, optional (default=None)
        Whether the selected column should be treated as a categorical or
        numerical feature. If set to ``None``, the type of the column will be
        inferred from the data therein. If set to ``False``, the column will be
        treated as numerical unless it is string-based in which case a warning
        will be emitted and the column will be treated as numerical despite
        this setting. Finally, if set to ``True``, the column will be treated
        as categorical.

    Warns
    -----
    UserWarning
        When grouping is done on a categorical column a warning is emitted when
        some of the values in that column are not accounted for, i.e. they are
        not included in the ``groupings`` parameter. Also, if some of the rows
        are not included in any of the groupings, a warning is shown. Missing
        row indices may be a result of some of the values being not-a-number
        for a numerical column and missing some of the unique values for a
        categorical column. ``treat_as_categorical`` parameter is set to
        ``False``, however the feature selected is string-based
        (i.e. categorical), therefore cannot be treated as a numerical one.

    Raises
    ------
    IncorrectShapeError
        The input ``dataset`` is not 2-dimensional.
    IndexError
        The supplied ``column_index`` is not valid for the input ``dataset``.
    TypeError
        The column index is neither a string nor an integer. The numerical bins
        number is not an integer. The ``groupings`` parameter is neither a list
        not ``None``. One of the grouping bin boundaries (for a numerical
        feature column) is not a number. One of the groupings (for a
        categorical feature column) is not a tuple. The
        ``treat_as_categorical`` parameter is neither a boolean nor ``None``.
    ValueError
        The input ``dataset`` is not of a base type. The numerical bins number
        is less than 2. The ``groupings`` list is empty. The numbers in the
        ``groupings`` parameter are not monotonically increasing (for a
        numerical column). There are duplicate values shared among tuples in
        the ``grouping`` parameter or one of the values does not appear in the
        selected column (for a categorical column).

    Returns
    -------
    indices_per_bin : List[List[integer]]
        A list of lists with the latter one holding row indices of a particular
        group.
    bin_names : List[string]
        A list holding a description of each group.
    """
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    if not fuav.is_2d_array(dataset):
        raise IncorrectShapeError('The input array should be 2-dimensional.')

    if not fuav.is_base_array(dataset):
        raise ValueError('The input array should be of a base type (a mixture '
                         'of numerical and textual types).')

    # Check index validity
    if isinstance(column_index, (str, int)):
        if not fuat.are_indices_valid(dataset, np.array([column_index])):
            raise IndexError('*{}* is not a valid column index for the input '
                             'dataset.'.format(column_index))
    else:
        raise TypeError('The column index can either be a string or an '
                        'integer.')

    # Check the number of numerical bins
    if isinstance(numerical_bins_number, int):
        if numerical_bins_number < 2:
            raise ValueError('The numerical_bins_number needs to be at least '
                             '2.')
    else:
        raise TypeError('The numerical_bins_number parameter has to be an '
                        'integer.')

    # Check treat_as_categorical
    if treat_as_categorical is not None:
        if not isinstance(treat_as_categorical, bool):
            raise TypeError('The treat_as_categorical parameter has to be a '
                            'boolean.')

    if fuav.is_structured_array(dataset):
        column = dataset[column_index]
    else:
        column = dataset[:, column_index]
    assert fuav.is_1d_array(column), 'This must be a 1D numpy array.'

    # Get a list of all the row indices
    all_row_indices = set(range(column.shape[0]))

    indices_per_bin = []
    bin_names = []

    is_numerical_column = fuav.is_numerical_array(column)
    is_categorical_column = fuav.is_textual_array(column)
    assert is_numerical_column is not is_categorical_column, \
        'The column must be a base array.'

    # Sort out numerical/categorical column treatment
    if treat_as_categorical is None:
        go_numerical = is_numerical_column
    else:
        if treat_as_categorical:
            go_numerical = False
        else:  # Treat as numerical
            if is_numerical_column:
                go_numerical = True
            else:  # Is not numerical
                warnings.warn(
                    'Selected feature is categorical, therefore cannot be '
                    'treated as numerical. The feature will be treated as '
                    'categorical despite the treat_as_categorical parameter '
                    'set to False.', UserWarning)
                go_numerical = False

    if go_numerical:
        if groupings is None:
            # Get default bins
            bins = np.linspace(
                column.min(),
                column.max(),
                num=numerical_bins_number,
                endpoint=False)[1:].tolist()
        elif isinstance(groupings, list):
            if not groupings:
                raise ValueError('A numerical grouping list has to contain at '
                                 'least one element.')

            # Every element in the groupings list must be a number
            for i, number in enumerate(groupings):
                if not isinstance(number, Number):
                    raise TypeError('For a numerical column all of the '
                                    'grouping items must be numbers. *{}* '
                                    'is not a number.'.format(number))
                if i != 0:
                    if number <= groupings[i - 1]:
                        raise ValueError('The numbers in the groupings list '
                                         'have to be monotonically '
                                         'increasing.')
            bins = groupings
        else:
            raise TypeError('Since a numerical column was chosen the grouping '
                            'must be a list of bin boundaries or None.')

        lower_edge = 'x <= {}'
        middle = '{} < x <= {}'
        upper_edge = '{} < x'

        indices_seen_so_far = set()  # type: Set[int]

        for i, edge in enumerate(bins):
            if i == 0:
                indices = np.where(column <= edge)[0].tolist()

                indices_per_bin.append(indices)
                bin_names.append(lower_edge.format(edge))
            else:
                edge_lower = bins[i - 1]

                indices_l = set(np.where(column <= edge)[0].tolist())
                indices_u = set(np.where(column > edge_lower)[0].tolist())
                indices = list(indices_l.intersection(indices_u))

                indices_per_bin.append(indices)
                bin_names.append(middle.format(edge_lower, edge))

            assert not indices_seen_so_far.intersection(indices), 'Duplicates.'
            indices_seen_so_far = indices_seen_so_far.union(indices)

        assert bins, 'If bins is empty, i and edge will not be defined.'
        # pylint: disable=undefined-loop-variable
        indices = np.where(column > edge)[0].tolist()

        indices_per_bin.append(indices)
        bin_names.append(upper_edge.format(edge))

        assert not indices_seen_so_far.intersection(indices), 'Duplicates.'
        indices_seen_so_far = indices_seen_so_far.union(indices)
    else:
        unique_elements = np.sort(np.unique(column)).tolist()

        if groupings is None:
            bins = [(i, ) for i in unique_elements]
        elif isinstance(groupings, list):
            if not groupings:
                raise ValueError('A categorical grouping list has to contain '
                                 'at least one element.')

            values_seen_so_far = set()  # type: Set[str]

            # Every element in the groupings list must be a valid tuple
            for value_tuple in groupings:
                if not isinstance(value_tuple, tuple):
                    raise TypeError('For a categorical column all of the '
                                    'grouping items must be tuples. *{}* '
                                    'is not a tuple.'.format(value_tuple))
                for value in value_tuple:
                    if value not in unique_elements:
                        raise ValueError('*{}* value is not present in the '
                                         'selected column.'.format(value))

                if values_seen_so_far.intersection(value_tuple):
                    raise ValueError('Some values are duplicated across '
                                     'tuples.')
                values_seen_so_far = values_seen_so_far.union(value_tuple)

            unaccounted_values = set(unique_elements).difference(
                values_seen_so_far)
            if unaccounted_values:
                warnings.warn(
                    'The following values in the selected column were not '
                    'accounted for in the grouping '
                    'tuples:\n{}.'.format(unaccounted_values), UserWarning)

            bins = [tuple(sorted(i)) for i in groupings]  # type: ignore
            bins = sorted(bins)
        else:
            raise TypeError('Since a categorical column was chosen the '
                            'grouping must be a list of tuples representing '
                            'categorical values grouping or None for the '
                            'default grouping.')

        indices_seen_so_far = set()

        for bin_values in bins:
            indices = set()
            for value in bin_values:
                vid = np.where(column == value)[0].tolist()
                indices = indices.union(vid)

            indices_per_bin.append(list(indices))
            bin_names.append('{}'.format(bin_values))

            assert not indices_seen_so_far.intersection(indices), 'Duplicates.'
            indices_seen_so_far = indices_seen_so_far.union(indices)

    # Validate that all of the row indices were accounted for
    missed_indices = all_row_indices.difference(indices_seen_so_far)
    if missed_indices:
        warnings.warn(
            'The following row indices could not be accounted for:\n{}.\n For '
            'a numerical column there may have been some numpy.nan therein. '
            'For a categorical column some of the column values were probably '
            'not specified in the grouping, in which case there should be a '
            'separate user warning.'.format(missed_indices), UserWarning)

    return indices_per_bin, bin_names


def apply_to_column_grouping(
        labels: np.ndarray, predictions: np.ndarray,
        row_grouping: List[List[int]],
        fnc: Callable[[np.ndarray, np.ndarray], float]) -> List[float]:
    """
    Applies a function to the specified groups of labels and predictions.

    This functions allows to apply a metric for a particular data grouping. The
    two main applications are group-based fairness and performance evaluation.

    Parameters
    ----------
    labels : numpy.ndarray
        A ground truth numpy array.
    predictions : numpy.ndarray
        A predictions numpy array.
    row_grouping : List[List[integer]]
        A list of lists representing row indices of the ground truth and
        prediction arrays resulting in their grouping.
    fnc : Callable[[numpy.ndarray, numpy.ndarray], number]
        A function (metric) that will be applied to all of the groups defined
        by the ``row_grouping`` parameter.

    Raises
    ------
    AttributeError
        The ``fnc`` parameter does not require two input parameters.
    IncorrectShapeError
        The ``labels`` or ``predictions`` parameter is not a 1-dimensional
        numpy array. The ``labels`` and ``predictions`` arrays are not of the
        same length.
    TypeError
        The ``row_grouping`` parameter is not a list. One of the elements of
        the ``row_grouping`` is not a list. Some of the elements in the inner
        list of the ``row_grouping`` list are not integers. The ``fnc``
        parameter is not a callable (function).
    ValueError
        The ``row_grouping`` parameter is an empty list. Some of the values in
        the ``row_grouping`` list are duplicated.

    Returns
    -------
    applied : List[numbers]
        A list with the ``fnc`` function result for every group defined by the
        ``row_grouping`` parameter.
    """
    # pylint: disable=too-many-branches
    if not fuav.is_1d_array(labels):
        raise IncorrectShapeError('The labels array should be 1-dimensional.')
    if not fuav.is_1d_array(predictions):
        raise IncorrectShapeError('The predictions array should be '
                                  '1-dimensional.')
    if labels.shape[0] != predictions.shape[0]:
        raise IncorrectShapeError('The labels and predictions arrays should '
                                  'be of the same length.')

    if isinstance(row_grouping, list):
        if not row_grouping:
            raise ValueError('The row_grouping parameter cannot be an empty '
                             'list.')
        duplicated_indices = set()  # type: Set[int]
        for i in row_grouping:
            if not isinstance(i, list):
                raise TypeError('All of the elements of the row_grouping list '
                                'have to be lists.')
            if not i:
                raise ValueError('All of the elements of the row_grouping '
                                 'list must be non-empty lists.')
            for j in i:
                if not isinstance(j, int):
                    raise TypeError('All of the elements of the inner lists '
                                    'in the row_grouping have to be integers.')
            if duplicated_indices.intersection(i):
                raise ValueError('Some of the values in the row_grouping are '
                                 'duplicated.')
            duplicated_indices = duplicated_indices.union(i)
    else:
        raise TypeError('The row_grouping parameter has to be a list.')

    if not callable(fnc):
        raise TypeError('The fnc parameter is not callable (a function).')
    required_param_n = 0
    params = inspect.signature(fnc).parameters
    for param in params:
        if params[param].default is params[param].empty:
            required_param_n += 1
    if required_param_n != 2:
        raise AttributeError('Provided function (fnc) does not require 2 '
                             'input parameters. The first required parameter '
                             'should be ground truth labels and the second '
                             'one predictions.')

    applied = [fnc(labels[grp], predictions[grp]) for grp in row_grouping]

    return applied


def validate_indices_per_bin(indices_per_bin: List[List[int]]) -> bool:
    """
    Validates a list of binned indices.

    Parameters
    ----------
    indices_per_bin : List[List[integer]]
        A list of lists with the latter one holding row indices of a particular
        group.

    Warns
    -----
    UserWarning
        Some of the indices may be missing -- gaps in the consecutive list of
        indices were found.

    Raises
    ------
    TypeError
        The ``indices_per_bin`` parameter is not a list, one of the elements of
        the ``indices_per_bin`` list is not a list or one of the elements of
        the latter list is not an integer.
    ValueError
        The ``indices_per_bin`` list is empty. Some of the indices are
        duplicated or are negative integers.

    Returns
    -------
    is_valid : boolean
        ``True`` if the input is valid, ``False`` otherwise.
    """
    # pylint: disable=too-many-nested-blocks
    is_valid = False

    flat_list = []  # type: List[int]
    if isinstance(indices_per_bin, list):
        if not indices_per_bin:
            raise ValueError('The indices_per_bin list cannot be empty.')
        for indices_bin in indices_per_bin:
            if isinstance(indices_bin, list):
                flat_list += indices_bin
                for index in indices_bin:
                    if isinstance(index, int):
                        if index < 0:
                            raise ValueError('One of the indices is a '
                                             'negative integer -- all should '
                                             'be non-negative.')
                    else:
                        raise TypeError('Indices should be integers. *{}* is '
                                        'not an integer.'.format(index))
            else:
                raise TypeError('One of the elements embedded in the '
                                'indices_per_bin list is not a list.')
        if len(flat_list) != len(set(flat_list)):
            raise ValueError('Some of the indices are duplicated.')
    else:
        raise TypeError('The indices_per_bin parameter has to be a list.')

    # Check whether the indices are consecutive numbers without any gaps
    indices_number = max(flat_list) + 1  # The indexing starts from 0
    all_indices = range(indices_number)
    missing_indices = set(all_indices).difference(flat_list)
    if missing_indices:
        warnings.warn(
            'The following indices are missing (based on the top index): {}.\n'
            'It is possible that more indices are missing if they were the '
            'last one(s).'.format(missing_indices), UserWarning)

    is_valid = True
    return is_valid


def validate_binary_matrix(binary_array: np.ndarray,
                           name: Optional[str] = None) -> bool:
    """
    Validates a binary, square and symmetric numpy array.

    Parameters
    ----------
    binary_array : numpy.ndarray
        A square (equal number of rows and columns), boolean  symmetric numpy
        array.

    Raises
    ------
    IncorrectShapeError
        The matrix is not 2-dimensional or square.
    TypeError
        The matrix is not of boolean type.
    ValueError
        The matrix is a structured numpy array or is not diagonally symmetric.

    Returns
    -------
    is_valid : boolean
        ``True`` if the matrix is valid, ``False`` otherwise.
    """
    if name is None:
        name = ''
    else:
        assert isinstance(name, str), 'The name parameter has to be string.'
        name = name.strip()
        name = '{} '.format(name) if name else name
    is_valid = False

    if not fuav.is_2d_array(binary_array):
        raise IncorrectShapeError('The {}matrix has to be '
                                  '2-dimensional.'.format(name))
    if fuav.is_structured_array(binary_array):
        raise ValueError('The {}matrix cannot be a structured numpy '
                         'array.'.format(name))
    if binary_array.dtype != bool:
        raise TypeError('The {}matrix has to be of boolean '
                        'type.'.format(name))
    if binary_array.shape[0] != binary_array.shape[1]:
        raise IncorrectShapeError('The {}matrix has to be '
                                  'square.'.format(name))
    if (not np.array_equal(binary_array, binary_array.T)
            or np.diagonal(binary_array).any()):
        raise ValueError('The {}matrix has to be diagonally '
                         'symmetric.'.format(name))

    is_valid = True
    return is_valid
