"""
The :mod:`fatf.fairness.models.measures` module holds models fairness measures.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
#         Rafael Poyiadzi <rp13102@bristol.ac.uk>
# License: BSD 3 clause

from numbers import Number
from typing import List, Optional, Tuple, Union

import numpy as np

import fatf.utils.data.tools as fudt
import fatf.utils.metrics.tools as fumt
import fatf.utils.metrics.subgroup_metrics as fums

__all__ = ['disparate_impact',
           'disparate_impact_indexed',
           'disparate_impact_check',
           'demographic_parity',
           'equal_opportunity',
           'equal_accuracy']  # yapf: disable

Index = Union[int, str]  # A column index type


def disparate_impact(
        dataset: np.ndarray,
        #
        ground_truth: np.ndarray,
        predictions: np.ndarray,
        #
        column_index: Index,
        #
        label_index: int = 0,
        #
        groupings: Optional[List[Union[float, Tuple[str]]]] = None,
        numerical_bins_number: int = 5,
        treat_as_categorical: Optional[bool] = None,
        #
        labels: Optional[List[Union[str, float]]] = None,
        #
        tolerance: float = 0.2,
        criterion: Optional[str] = None) -> Tuple[np.ndarray, List[str]]:
    """
    Calculates selected disparate impact grid for a data set.

    This function combines
    :func:`fatf.utils.metrics.tools.confusion_matrix_per_subgroup` function
    together with :func:`fatf.utils.metrics.subgroup_metrics.apply_metric`
    function. For the description of parameters, errors and exceptions please
    see the documentation of these functions.

    Parameters
    ----------
    dataset, ground_truth, predictions, column_index, groupings, \
numerical_bins_number, labels, and treat_as_categorical
        See the documentation of
        :func:`fatf.utils.metrics.tools.confusion_matrix_per_subgroup`
        function.
    label_index : integer
        The index of the "positive" class in the confusion matrix. (Not
        required for binary problems.) See the description of
        :func:`fatf.utils.data.tools.group_by_column` function.
    criterion : Union[None, string]
        A string representing group fairness criterion. One of:
        ``'demographic parity'``, ``'equal opportunity'``, ``'equal accuracy'``
        or ``None`` for the default option ``'equal accuracy'``.
    tolerance : number
        A number between 0 and 1 that indicates how much any two metrics can
        differ to be considered "equal".

    Raises
    ------
    TypeError
        The ``criterion`` parameter is neither ``None`` nor a string.
    ValueError
        The ``criterion`` parameter is none of the allowed values (see the
        description of the parameter).

    Returns
    -------
    disparity_grid : numpy.ndarray
        A square, symmetric, boolean numpy array that indicates for which pair
        of sub-populations a disparity happens.
    """
    # pylint: disable=too-many-arguments
    population_cmxs, bin_names = fumt.confusion_matrix_per_subgroup(
        dataset, ground_truth, predictions, column_index, groupings,
        numerical_bins_number, treat_as_categorical, labels)

    disparity_grid = _disparate_impact_grid(population_cmxs, criterion,
                                            tolerance, label_index)

    return disparity_grid, bin_names


def disparate_impact_indexed(
        indices_per_bin: List[np.ndarray],
        ground_truth: np.ndarray,
        predictions: np.ndarray,
        #
        label_index: int = 0,
        #
        labels: Optional[List[Union[str, float]]] = None,
        #
        tolerance: float = 0.2,
        criterion: Optional[str] = None) -> np.ndarray:
    """
    Calculates selected disparate impact grid for indexed data.

    This function combines
    :func:`fatf.utils.metrics.tools.confusion_matrix_per_subgroup_indexed`
    function together with
    :func:`fatf.utils.metrics.subgroup_metrics.apply_metric` function. For the
    description of parameters, errors and exceptions please see the
    documentation of these functions.

    Parameters
    ----------
    indices_per_bin, ground_truth, predictions, and labels
        See the documentation of
        :func:`fatf.utils.metrics.tools.confusion_matrix_per_subgroup_indexed`
        function.
    label_index : integer
        The index of the "positive" class in the confusion matrix. (Not
        required for binary problems.) See the description of
        :func:`fatf.utils.data.tools.group_by_column` function.
    criterion : Union[None, string]
        A string representing group fairness criterion. One of:
        ``'demographic parity'``, ``'equal opportunity'``, ``'equal accuracy'``
        or ``None`` for the default option ``'equal accuracy'``.
    tolerance : number
        A number between 0 and 1 that indicates how much any two metrics can
        differ to be considered "equal".

    Raises
    ------
    TypeError
        The ``criterion`` parameter is neither ``None`` nor a string.
    ValueError
        The ``criterion`` parameter is none of the allowed values (see the
        description of the parameter).

    Returns
    -------
    disparity_grid : numpy.ndarray
        A square, symmetric, boolean numpy array that indicates for which pair
        of sub-populations a disparity happens.
    """
    # pylint: disable=too-many-arguments
    population_cmxs = fumt.confusion_matrix_per_subgroup_indexed(
        indices_per_bin, ground_truth, predictions, labels)

    disparity_grid = _disparate_impact_grid(population_cmxs, criterion,
                                            tolerance, label_index)

    return disparity_grid


def _disparate_impact_grid(confusion_matrix_list: List[np.ndarray],
                           criterion: Union[None, str], tolerance: float,
                           label_index: int) -> np.ndarray:
    """
    Applies selected group fairness criterion to get a disparate impact grid.

    Parameters
    ----------
    confusion_matrix_list : List[numpy.ndarray]
        A list of confusion matrices, one for each sub-population.
    criterion : Union[None, string]
        A string representing group fairness criterion. One of:
        ``'demographic parity'``, ``'equal opportunity'``, ``'equal accuracy'``
        or ``None`` for the default option ``'equal accuracy'``.
    tolerance : number
        A number between 0 and 1 that indicates how much any two metrics can
        differ to be considered "equal".
    label_index : integer
        The index of the "positive" class in the confusion matrix. (Not
        required for binary problems.)

    Raises
    ------
    TypeError
        The ``criterion`` parameter is neither ``None`` nor a string.
    ValueError
        The ``criterion`` parameter is none of the allowed values (see the
        description of the parameter).

    Returns
    -------
    disparity_grid : numpy.ndarray
        A square, symmetric, boolean numpy array that indicates for which pair
        of sub-populations a disparity happens.
    """
    criteria = ['demographic parity', 'equal opportunity', 'equal accuracy']
    if criterion is None:
        criterion = 'equal accuracy'
    elif isinstance(criterion, str):
        if criterion not in criteria:
            raise ValueError('Unrecognised criterion. The following options '
                             'are allowed: {}.'.format(criteria))
    else:
        raise TypeError('Criterion has to either be a string indicating '
                        'parity metric or None for the default parity metric '
                        '(equal accuracy).')

    if criterion == 'demographic parity':
        disparity_grid = demographic_parity(
            confusion_matrix_list,
            tolerance=tolerance,
            label_index=label_index)
    elif criterion == 'equal opportunity':
        disparity_grid = equal_opportunity(
            confusion_matrix_list,
            tolerance=tolerance,
            label_index=label_index)
    elif criterion == 'equal accuracy':
        disparity_grid = equal_accuracy(
            confusion_matrix_list, tolerance=tolerance)
    else:
        assert False, 'Unknown criterion.'  # pragma: nocover

    return disparity_grid


def disparate_impact_check(disparity_grid: np.ndarray) -> bool:
    """
    Checks if any sub-population pair violates chosen disparate impact measure.

    Parameters
    ----------
    disparity_grid : numpy.ndarray
        A square, diagonally symmetric, boolean numpy array representing
        pair-wise disparity between subpopulations. See the return value of
        :func:`fatf.fairness.models.measures.demographic_parity` as an example.

    Raises
    ------
    IncorrectShapeError
        The disparity grid matrix is not 2-dimensional or square.
    TypeError
        The disparity grid matrix is not of boolean type.
    ValueError
        The disparity grid matrix is a structured numpy array or is not
        diagonally symmetric.

    Returns
    -------
    is_disparate : boolean
        ``True`` if there is a disparity between any pair of sub-populations,
        ``False`` otherwise.
    """
    assert fudt.validate_binary_matrix(disparity_grid,
                                       'disparate impact'), 'Invalid matrix.'
    is_disparate = disparity_grid.any()
    return is_disparate


def demographic_parity(confusion_matrix_list: List[np.ndarray],
                       tolerance: float = 0.2,
                       label_index: int = 0) -> np.ndarray:
    """
    Checks for demographic parity between all of the sub-populations.

    This function checks if **predictive positive rate** difference of all
    grouping pairs is within the tolerance level.

    .. note::
       This function expects a list of confusion matrices per sub-group for
       tested data. To get this list please use either
       :func:`fatf.utils.metrics.tools.confusion_matrix_per_subgroup` or
       :func:`fatf.utils.metrics.tools.confusion_matrix_per_subgroup_indexed`
       function.

       Alternatively you can call either
       :func:`fatf.fairness.models.measures.disparate_impact` or
       :func:`fatf.fairness.models.measures.disparate_impact_indexed` function,
       which handles both the grouping and calculates the desired
       group-fairness criterion.

    Parameters
    ----------
    confusion_matrix_list : List[numpy.ndarray]
        A list of confusion matrices, one for each sub-population.
    tolerance : number, optional (default=0.2)
        A number between 0 and 1 that indicates how much any two predictive
        positive rates can differ to be considered "equal".
    label_index : integer, optional (default=0)
        The index of the "positive" class in the confusion matrix. (Not
        required for binary problems.)

    Raises
    ------
    TypeError
        The tolerance parameter is not a number.
    ValueError
        The tolerance parameter is out of [0, 1] range.

    Returns
    -------
    disparity : numpy.ndarray
        A square and diagonally symmetric numpy array with boolean values.
        An entry is ``True`` if a pair of two sub-populations' predictive
        positive rate difference is above the tolerance level and ``False``
        otherwise.
    """
    assert _validate_tolerance(tolerance), 'Invalid tolerance parameter.'

    ppr = np.asarray(
        fums.apply_metric(
            confusion_matrix_list,
            'positive predictive value',
            label_index=label_index))
    disparity = np.abs(ppr[:, np.newaxis] - ppr[np.newaxis, :])
    disparity = disparity > tolerance

    assert np.array_equal(disparity, disparity.T), 'Must be symmetric.'

    return disparity


def equal_opportunity(confusion_matrix_list: List[np.ndarray],
                      tolerance: float = 0.2,
                      label_index: int = 0) -> np.ndarray:
    """
    Checks for equal opportunity between all of the sub-populations.

    This function checks if **true positive rate** difference of all grouping
    pairs is within the tolerance level.

    .. note::
       This function expects a list of confusion matrices per sub-group for
       tested data. To get this list please use either
       :func:`fatf.utils.metrics.tools.confusion_matrix_per_subgroup` or
       :func:`fatf.utils.metrics.tools.confusion_matrix_per_subgroup_indexed`
       function.

       Alternatively you can call either
       :func:`fatf.fairness.models.measures.disparate_impact` or
       :func:`fatf.fairness.models.measures.disparate_impact_indexed` function,
       which handles both the grouping and calculates the desired
       group-fairness criterion.

    Parameters
    ----------
    confusion_matrix_list : List[numpy.ndarray]
        A list of confusion matrices, one for each sub-population.
    tolerance : number, optional (default=0.2)
        A number between 0 and 1 that indicates how much any two true positive
        rates can differ to be considered "equal".
    label_index : integer, optional (default=0)
        The index of the "positive" class in the confusion matrix. (Not
        required for binary problems.)

    Raises
    ------
    TypeError
        The tolerance parameter is not a number.
    ValueError
        The tolerance parameter is out of [0, 1] range.

    Returns
    -------
    disparity : numpy.ndarray
        A square and diagonally symmetric numpy array with boolean values.
        An entry is ``True`` if a pair of two sub-populations' true positive
        rate difference is above the tolerance level and ``False`` otherwise.
    """
    assert _validate_tolerance(tolerance), 'Invalid tolerance parameter.'

    ppr = np.asarray(
        fums.apply_metric(
            confusion_matrix_list,
            'true positive rate',
            label_index=label_index))
    disparity = np.abs(ppr[:, np.newaxis] - ppr[np.newaxis, :])
    disparity = disparity > tolerance

    assert np.array_equal(disparity, disparity.T), 'Must be symmetric.'

    return disparity


def equal_accuracy(confusion_matrix_list: List[np.ndarray],
                   tolerance: float = 0.2,
                   label_index: int = 0) -> np.ndarray:
    """
    Checks if accuracy difference of all grouping pairs is within tolerance.

    .. note::
       This function expects a list of confusion matrices per sub-group for
       tested data. To get this list please use either
       :func:`fatf.utils.metrics.tools.confusion_matrix_per_subgroup` or
       :func:`fatf.utils.metrics.tools.confusion_matrix_per_subgroup_indexed`
       function.

       Alternatively you can call either
       :func:`fatf.fairness.models.measures.disparate_impact` or
       :func:`fatf.fairness.models.measures.disparate_impact_indexed` function,
       which handles both the grouping and calculates the desired
       group-fairness criterion.

    Parameters
    ----------
    confusion_matrix_list : List[numpy.ndarray]
        A list of confusion matrices, one for each sub-population.
    tolerance : number, optional (default=0.2)
        A number between 0 and 1 that indicates how much any two accuracies can
        differ to be considered "equal".
    label_index : integer, optional (default=0)
        The index of the "positive" class in the confusion matrix. (Not
        required for binary problems.)

    Raises
    ------
    TypeError
        The tolerance parameter is not a number.
    ValueError
        The tolerance parameter is out of [0, 1] range.

    Returns
    -------
    disparity : numpy.ndarray
        A square and diagonally symmetric numpy array with boolean values.
        An entry is ``True`` if a pair of two sub-populations' accuracy
        difference is above the tolerance level and ``False`` otherwise.
    """
    assert _validate_tolerance(tolerance), 'Invalid tolerance parameter.'

    ppr = np.asarray(
        fums.apply_metric(
            confusion_matrix_list, 'accuracy', label_index=label_index))
    disparity = np.abs(ppr[:, np.newaxis] - ppr[np.newaxis, :])
    disparity = disparity > tolerance

    assert np.array_equal(disparity, disparity.T), 'Must be symmetric.'

    return disparity


def _validate_tolerance(tolerance: float) -> bool:
    """
    Validate a tolerance parameter.

    Parameters
    ----------
    tolerance : number
        A number representing disparity tolerance.

    Raises
    ------
    TypeError
        The tolerance parameter is not a number.
    ValueError
        The tolerance parameter is out of [0, 1] range.

    Returns
    -------
    is_valid : boolean
        ``True`` if the tolerance parameter is valid, ``False`` otherwise.
    """
    is_valid = False

    if isinstance(tolerance, Number):
        if tolerance < 0 or tolerance > 1:
            raise ValueError('The tolerance parameter should be within [0, 1] '
                             'range.')
    else:
        raise TypeError('The tolerance parameter should be a number.')

    is_valid = True
    return is_valid
