"""
The :mod:`fatf.accountability.data.measures` module holds data accountability
measures.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
#         Rafael Poyiadzi <rp13102@bristol.ac.uk>
# License: new BSD

from numbers import Number
from typing import List, Optional, Tuple, Union

import numpy as np

import fatf.utils.data.tools as fudt

__all__ = ['sampling_bias',
           'sampling_bias_indexed',
           'sampling_bias_grid_check',
           'sampling_bias_check']  # yapf: disable

Index = Union[int, str]  # A column index type


def sampling_bias(dataset: np.ndarray,
                  column_index: Index,
                  groupings: Optional[List[Union[float, Tuple[str]]]] = None,
                  numerical_bins_number: int = 5,
                  treat_as_categorical: Optional[bool] = None
                  ) -> Tuple[List[int], np.ndarray, List[str]]:
    """
    Computes information needed for evaluating and remedying sampling bias.

    Computes the *number of instances* per sub-population defined by the input
    parameters, the *weights* that can be used for cost-sensitive learning to
    mitigate the sampling bias and the *names* of each sub-population (in terms
    of the selected feature and its values).

    .. note::
       To evaluate the sampling bias in terms of a binary ``True``/``False``
       answer please use the
       :func:`fatf.accountability.data.measures.sampling_bias_check` function
       or :func:`fatf.accountability.data.measures.sampling_bias_grid_check`
       function to see sub-population pairwise sampling bias.

    For warnings raised by this method please see the documentation of
    :func:`fatf.utils.data.tools.validate_indices_per_bin` function.

    Parameters
    ----------
    dataset, column_index, groupings, numerical_bins_number, and \
treat_as_categorical
        These parameters are described in the documentation of
        :func:`fatf.utils.data.tools.group_by_column` function and are used to
        define a grouping (i.e. sub-populations). If you have your own
        index-based grouping and would like to get counts and weights for
        cost-sensitive learning, please consider using
        :func:`fatf.accountability.data.measures.sampling_bias_indexed`
        function.

    Returns
    -------
    counts : List[integers]
        A number of data points for each sub-population defined by partitioning
        of the selected feature.
    weights : numpy.ndarray
        A weight for every instance (that could be grouped, i.e. assigned to
        one of the sub-populations) in the input ``dataset``. The weights are
        useful for training a cost-sensitive classifier to mitigate the
        sampling bias. The weights are inversely proportional to the number of
        instance occurrences for every sub-population.
    bin_names : List[strings]
        The name of every sub-population (binning results) defined by the
        feature ranges for a numerical feature and feature value sets for a
        categorical feature.
    """
    indices_per_bin, bin_names = fudt.group_by_column(
        dataset, column_index, groupings, numerical_bins_number,
        treat_as_categorical)

    assert fudt.validate_indices_per_bin(indices_per_bin), \
        'Binned indices list is invalid.'

    counts = [len(i) for i in indices_per_bin]
    weights = _get_weights(indices_per_bin)
    return counts, weights, bin_names


def sampling_bias_indexed(
        indices_per_bin: List[List[int]]) -> Tuple[List[int], np.ndarray]:
    """
    Computes information needed for evaluating and remedying sampling bias.

    Computes the *number of instances* per sub-population based on the number
    of indices per sub-population and the *weights* that can be used for
    cost-sensitive learning to mitigate the sampling bias.

    This is an alternative to
    :func:`fatf.accountability.data.measures.sampling_bias` function, which can
    be used when one already has the desired instance binning.

    For warnings and errors raised by this method please see the documentation
    of :func:`fatf.utils.data.tools.validate_indices_per_bin` function.

    Parameters
    ----------
    indices_per_bin : List[List[integer]]
        A list of lists with the latter one holding row indices of a particular
        group (sub-population).

    Returns
    -------
    counts : List[integers]
        A number of data points for each sub-population defined by partitioning
        of the selected feature.
    weights : numpy.ndarray
        A weight for every instance (that could be grouped, i.e. assigned to
        one of the sub-populations) in the input ``dataset``. The weights are
        useful for training a cost-sensitive classifier to mitigate the
        sampling bias. The weights are inversely proportional to the number of
        instance occurrences for every sub-population.
    """
    assert fudt.validate_indices_per_bin(indices_per_bin), \
        'Binned indices list is invalid.'

    counts = [len(i) for i in indices_per_bin]
    weights = _get_weights(indices_per_bin)
    return counts, weights


def sampling_bias_grid_check(counts: List[int],
                             threshold: float = 0.8) -> np.ndarray:
    """
    Checks for a pairwise sampling bias based on the provided threshold.

    This functions checks the two-way (x/y and y/x proportion of counts) and
    if any of them (the absolute value of the proportion-1, to be precise) is
    below the threshold the given pair is considered to be suffering form a
    sampling bias.

    Parameters
    ----------
    counts : List[integer]
        A list of integers representing the number of instances in every
        sub-group.
    threshold : float, optional (default=0.8)
        A threshold (number between 0 and 1 inclusive) that defines when a
        sampling bias occurs.

    Raises
    ------
    TypeError
        The ``counts`` parameter is not a list or one of the elements of this
        list is not an integer. The ``threshold`` parameter is not a number.
    ValueError
        One of the counts is a negative integer. The ``threshold`` parameter
        out of range; the allowed range is 0 to 1 inclusive.

    Returns
    -------
    grid_check : numpy.ndarray
        A square (with the width and height being the number of sub-groups
        defined by the ``counts`` parameter) symmetric boolean numpy arrays
        with ``True`` for every sub-group pair that violates the systematic
        bias threshold. The order of rows and columns correspond to the order
        of sub-groups in the ``counts`` parameter.
    """
    assert _validate_counts(counts), 'Invalid counts parameter.'
    assert _validate_threshold(threshold), 'Invalid threshold parameter.'

    counts_array = np.asarray(counts)
    inv_threshold = 1 - threshold
    # Get pairwise proportions
    proportions = counts_array[np.newaxis, :] / counts_array[:, np.newaxis]
    proportions = np.abs(proportions - 1)

    # Check if any pair differs by more than the threshold
    grid_check = proportions > inv_threshold
    np.logical_or(grid_check, grid_check.T, out=grid_check)

    return grid_check


def sampling_bias_check(counts: List[int], threshold: float = 0.8) -> bool:
    """
    Checks for a pairwise sampling bias based on the provided threshold.

    Please see the documentation of
    :func:`fatf.accountability.data.measures.sampling_bias_grid_check` function
    for a description of input parameters, errors and exceptions.

    Returns
    -------
    is_biased : boolean
        ``True`` if any sub-group pair does not satisfy the sampling bias
        threshold, ``False`` otherwise.
    """
    grid_check = sampling_bias_grid_check(counts, threshold)
    is_biased = grid_check.any()
    return is_biased


def _validate_counts(counts: List[int]) -> bool:
    """
    Validates the counts parameter.

    Parameters
    ----------
    counts : List[integers]
        A list of counts.

    Raises
    ------
    TypeError
        The ``counts`` parameter is not a list or one of the elements of this
        list is not an integer.
    ValueError
        One of the counts is a negative integer.

    Returns
    -------
    is_valid : boolean
        ``True`` if counts is valid, ``False`` otherwise.
    """
    is_valid = False

    if isinstance(counts, list):
        for count in counts:
            if isinstance(count, int):
                if count < 0:
                    raise ValueError('Counts cannot be negative integers.')
            else:
                raise TypeError('Counts have to be integers.')
    else:
        raise TypeError('The counts parameter has to be a list of integers.')

    is_valid = True
    return is_valid


def _validate_threshold(threshold: float) -> bool:
    """
    Validates the threshold parameter.

    Parameters
    ----------
    threshold : number
        A threshold between 0 and 1.

    Raises
    ------
    TypeError
        The ``threshold`` parameter is not a number.
    ValueError
        The ``threshold`` parameter out of range; the allowed range is 0 to 1
        inclusive.

    Returns
    -------
    is_valid : boolean
        ``True`` if threshold is valid, ``False`` otherwise.
    """
    is_valid = False

    if isinstance(threshold, Number):
        if threshold < 0 or threshold > 1:
            raise ValueError('The threshold should be between 0 and 1 '
                             'inclusive.')
    else:
        raise TypeError('The threshold parameter has to be a number.')

    is_valid = True
    return is_valid


def _get_weights(indices_per_bin: List[List[int]]) -> np.ndarray:
    """
    Computes a weight for the binned instances to counteract the sampling bias.

    The weights can be used, for example, for cost-sensitive learning. The
    weights are computed in a way such that the sum of weights for all of the
    items in every sub-group is equal and the weights for all of the instances
    across all the sub-groups sum up to 1.

    If some of the indices are missing (the indices do not form a series of
    consecutive numbers the weight for that index is assign ``numpy.nan``.

    Parameters
    ----------
    indices_per_bin : List[List[integer]]
        A list of lists with the latter one holding row indices of a particular
        group (sub-population).

    Returns
    -------
    weights : numpy.ndarray
        An array of weights, one for each instance.
    """
    assert isinstance(indices_per_bin, list), 'Must be a list.'
    flat_list = []  # type: List[int]
    for indices_bin in indices_per_bin:
        assert isinstance(indices_bin, list), 'Must be a list.'
        flat_list += indices_bin
        for index in indices_bin:
            assert isinstance(index, int), 'Must be an integer.'
            assert index >= 0, 'Must be a positive integer.'
    assert len(flat_list) == len(set(flat_list)), 'Must not have duplicates.'

    indices_number = max(flat_list) + 1  # The indexing starts from 0
    groups_number = len(indices_per_bin)
    scales = [(1 / len(i)) / groups_number for i in indices_per_bin]

    weights = np.array(indices_number * [np.nan])
    for i, bin_indices in enumerate(indices_per_bin):
        weights[bin_indices] = scales[i]

    return weights
