"""
Implements data accountability measures.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
#         Rafael Poyiadzi <rp13102@bristol.ac.uk>
# License: new BSD

import inspect

from numbers import Number
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from fatf.exceptions import IncorrectShapeError

import fatf.utils.array.validation as fuav
import fatf.utils.data.tools as fudt
import fatf.utils.models.metrics as fumm

__all__ = ['systematic_performance_bias',
           'systematic_performance_bias_indexed',
           'apply_metric_function',
           'apply_metric',
           'systematic_performance_bias_grid_check',
           'systematic_performance_bias_check']  # yapf: disable

Index = Union[int, str]  # A column index type


def systematic_performance_bias(
        dataset: np.ndarray,
        #
        ground_truth: np.ndarray,
        predictions: np.ndarray,
        #
        column_index: Index,
        groupings: Optional[List[Union[Number, Tuple[str]]]] = None,
        numerical_bins_number: int = 5,
        #
        labels: Optional[List[Union[str, Number]]] = None
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Computes confusion matrices for every defined sub-population.

    This is useful for computing a variety of performance metrics for each
    sub-population, hence is a vital part of systematic performance bias
    evaluation.

    .. note::
       To evaluate the performance bias in terms of a binary ``True``/``False``
       answer please use the :func:`fatf.accountability.models.measures.
       systematic_performance_bias_check` function.

    For warnings raised by this method please see the documentation of
    :func:`fatf.utils.data.tools.validate_indices_per_bin` function.

    Parameters
    ----------
    dataset, column_index, groupings, and numerical_bins_number
        These parameters are described in the documentation of
        :func:`fatf.utils.data.tools.group_by_column` function and are used to
        define a grouping (i.e. sub-populations). If you have your own
        index-based grouping and would like to get sub-population-based
        confusion matrices, please consider using :func:`fatf.accountability.
        models.measures.systematic_performance_bias_indexed` function.
    ground_truth, predictions, and labels
        These parameters are described in the documentation of
        :func:`fatf.utils.models.metrics.get_confusion_matrix` function and are
        used to calculate confusion matrices.

    Returns
    -------
    population_confusion_matrix : List[numpy.ndarray]
        A list of confusion matrices for each sub-population.
    bin_names : List[strings]
        The name of every sub-population (binning results) defined by the
        feature ranges for a numerical feature and feature value sets for a
        categorical feature.
    """
    indices_per_bin, bin_names = fudt.group_by_column(
        dataset, column_index, groupings, numerical_bins_number)

    assert fudt.validate_indices_per_bin(indices_per_bin), \
        'Binned indices list is invalid.'

    population_confusion_matrix = systematic_performance_bias_indexed(
        indices_per_bin, ground_truth, predictions, labels)
    return population_confusion_matrix, bin_names


def systematic_performance_bias_indexed(
        indices_per_bin: List[np.ndarray],
        ground_truth: np.ndarray,
        predictions: np.ndarray,
        labels: Optional[List[Union[str, Number]]] = None) -> List[np.ndarray]:
    """
    Computes confusion matrices for every defined sub-population.

    This is useful for computing a variety of performance metrics based on
    predefined instance index binning for each sub-population, hence is a vital
    part of systematic performance bias evaluation.

    This is an alternative to
    :func:`fatf.accountability.models.measures.systematic_performance_bias`
    function, which can be used when one already has the desired instance
    binning.

    For warnings and errors raised by this method please see the documentation
    of :func:`fatf.utils.data.tools.validate_indices_per_bin` function.

    Parameters
    ----------
    indices_per_bin : List[List[integer]]
        A list of lists with the latter one holding row indices of a particular
        group (sub-population).
    ground_truth, predictions, and labels
        These parameters are described in the documentation of
        :func:`fatf.utils.models.metrics.get_confusion_matrix` function and are
        used to calculate confusion matrices.

    Returns
    -------
    population_confusion_matrix : List[numpy.ndarray]
        A list of confusion matrices for each sub-population.
    """
    assert fudt.validate_indices_per_bin(indices_per_bin), \
        'Binned indices list is invalid.'

    if labels is None:
        if not fuav.is_1d_array(ground_truth):
            raise IncorrectShapeError('The ground_truth parameter should be a '
                                      '1-dimensional numpy array.')
        if not fuav.is_1d_array(predictions):
            raise IncorrectShapeError('The predictions parameter should be a '
                                      '1-dimensional numpy array.')
        labels = np.sort(np.unique(np.concatenate(
            [ground_truth, predictions]))).tolist()

    population_confusion_matrix = []
    for bin_indices in indices_per_bin:
        confusion_matrix = fumm.get_confusion_matrix(
            ground_truth[bin_indices], predictions[bin_indices], labels)
        population_confusion_matrix.append(confusion_matrix)
    return population_confusion_matrix


def apply_metric_function(population_confusion_matrix: List[np.ndarray],
                          metric_function: Callable[[np.ndarray], Number],
                          *args, **kwargs) -> List[Number]:
    """
    Applies the provided performance metric to every confusion matrix.

    The performance metric function needs to take a numpy.ndarray confusion
    matrix as the first parameter followed by any number of unnamed and named
    parameters provided by ``*args`` and ``**kwargs`` parameters.

    Parameters
    ----------
    population_confusion_matrix : List[numpy.ndarray]
        A list of confusion matrices for each sub-population.
    metric_function : Callable[[numpy.ndarray, *args, **kwargs], Number]
        A metric function that takes a confusion matrix as a first parameter,
        followed by any number of unnamed parameters (``*args``) and any number
        of named parameters (``**kwargs``) and outputs a single number -- the
        metric value.
    *args
        Unnamed arguments passed to the metric function.
    **kwargs
        Named arguments passed to the metric function.

    Raises
    ------
    AttributeError
        The ``metric_function`` parameter does not require at least one unnamed
        parameter.
    IncorrectShapeError
        The confusion matrix is not a 2-dimensional numpy array, it is not
        square (equal width and height) or its dimension is not at least 2x2.
    TypeError
        The confusion matrix is not of an integer kind (e.g. ``int``,
        ``numpy.int32``, ``numpy.int64``). One of the ``metric_function``
        outputs is not numerical. The ``metric_function`` is not Python
        callable. The ``population_confusion_matrix`` is not a list.
    ValueError
        The confusion matrix is a structured numpy array. The
        ``population_confusion_matrix`` parameter is an empty list.

    Returns
    -------
    metrics : List[numbers]
        A list with the value of the selected metric for every sub-population.
    """
    # Validate the confusion matrices type
    if isinstance(population_confusion_matrix, list):
        if not population_confusion_matrix:
            raise ValueError('The population_confusion_matrix parameter '
                             'cannot be an empty list.')
        for confusion_matrix in population_confusion_matrix:
            assert fumm.validate_confusion_matrix(confusion_matrix), \
                'Invalid confusion matrix.'
    else:
        raise TypeError('The population_confusion_matrix parameter has to be '
                        'a list.')
    # Validate metric_function
    if callable(metric_function):
        required_param_n = 0
        params = inspect.signature(metric_function).parameters
        for param in params:
            if params[param].default is params[param].empty:
                required_param_n += 1
        if not required_param_n:
            raise AttributeError('The metric_function callable needs to have '
                                 'at least one required parameter taking a '
                                 'confusion matrix. 0 were found.')
    else:
        raise TypeError('The metric_function parameter has to be a Python '
                        'callable.')

    metrics = []
    for cmx in population_confusion_matrix:
        metrics.append(metric_function(cmx, *args, **kwargs))

    for metric_value in metrics:
        if not isinstance(metric_value, Number):
            raise TypeError('One of the metric function outputs is not a '
                            'number: *{}*.'.format(metric_value))

    return metrics


def apply_metric(population_confusion_matrix: List[np.ndarray],
                 metric: Optional[str] = None,
                 label_index: int = 0, **kwargs) -> List[Number]:
    """
    Applies one of the predefined performance metric to all confusion matrices.

    Available metrics are:

    * ``true positive rate``,
    * ``true negative rate``,
    * ``false positive rate``,
    * ``false negative rate``,
    * ``positive predictive value``,
    * ``negative predictive value``,
    * ``accuracy``, and
    * ``treatment``.

    Parameters
    ----------
    population_confusion_matrix : List[numpy.ndarray]
        A list of confusion matrices for each sub-population.
    metric : string, optional (default='accuracy')
        A performance metric identifier that will be used.
    label_index : integer, optional (default=0)
        The index of a label that should be treated as "positive". All the
        other labels will be treated as "negative". This is only useful when
        the confusion matrices are multi-class.

    Raises
    ------
    TypeError
        The ``metric`` parameter is not a string.
    ValueError
        The ``metric`` parameter specifies an unknown metric.

    Returns
    -------
    metrics : List[number]
        A list with the value of the selected metric for every sub-population.
    """
    available_metrics = {
        'true positive rate': fumm.multiclass_true_positive_rate,
        'true negative rate': fumm.multiclass_true_negative_rate,
        'false positive rate': fumm.multiclass_false_positive_rate,
        'false negative rate': fumm.multiclass_false_negative_rate,
        'positive predictive value': fumm.multiclass_positive_predictive_value,
        'negative predictive value': fumm.multiclass_negative_predictive_value,
        'accuracy': fumm.accuracy,
        'treatment': fumm.multiclass_treatment
    }

    if metric is None:
        metric = 'accuracy'
    elif isinstance(metric, str):
        if metric not in available_metrics:
            available_metrics_names = sorted(list(available_metrics.keys()))
            raise ValueError('The selected metric (*{}*) is not recognised. '
                             'The following options are available: '
                             '{}.'.format(metric, available_metrics_names))
    else:
        raise TypeError('The metric parameter has to be a string.')

    if metric == 'accuracy':
        metrics = apply_metric_function(population_confusion_matrix,
                                        available_metrics[metric],
                                        **kwargs)
    else:
        metrics = apply_metric_function(population_confusion_matrix,
                                        available_metrics[metric],
                                        label_index,
                                        **kwargs)

    return metrics


def systematic_performance_bias_grid_check(
        metrics_list: List[Number],
        threshold: Number = 0.8) -> np.ndarray:
    """
    Checks for pairwise systematic bias in group-wise predictive performance.

    If a disparity in performance is found to be above the specified
    ``threshold`` a given pair sub-population performance metrics is considered
    biased.

    Parameters
    ----------
    metrics_list : List[Number]
        A list of predictive performance measurements for each sub-population.
    threshold : number, optional (default=0.8)
        A threshold (between 0 and 1) that defines performance disparity.

    Raises
    ------
    TypeError
        The ``metrics_list`` is not a list. One of the metric values in the
        ``metrics_list`` is not a number. The ``threshold`` is not a number.
    ValueError
        The ``metrics_list`` is an empty list. The threshold is out of 0 to 1
        range.

    Returns
    -------
    grid_check : np.ndarray
        A symmetric and square boolean numpy ndarray that indicates (``True``)
        whether any pair of sub-populations has significantly different
        predictive performance.
    """
    # Validate metrics list
    if isinstance(metrics_list, list):
        if not metrics_list:
            raise ValueError('The metrics list cannot be an empty list.')
        for metric_value in metrics_list:
            if not isinstance(metric_value, Number):
                raise TypeError('All metric values in the metrics_list should '
                                'be numbers.')
    else:
        raise TypeError('The metrics_list parameter has to be a list.')
    # Validate threshold
    if isinstance(threshold, Number):
        if threshold < 0 or threshold > 1:
            raise ValueError('The threshold should be between 0 and 1 '
                             'inclusive.')
    else:
        raise TypeError('The threshold parameter has to be a number.')

    metrics_array = np.asarray(metrics_list)
    inv_threshold = 1 - threshold
    # Get pairwise proportions
    proportions = metrics_array[np.newaxis, :] / metrics_array[:, np.newaxis]
    proportions = np.abs(proportions - 1)

    # Check if any pair differs by more than the threshold
    grid_check = proportions > inv_threshold
    grid_check = np.logical_or(grid_check, grid_check.T)

    return grid_check


def systematic_performance_bias_check(metrics_list: List[Number],
                                      threshold: Number = 0.8) -> bool:
    """
    Checks for a systematic bias in provided predictive performance values.

    Please see the documentation of :func:`fatf.accountability.models.measures.
    systematic_performance_bias_grid_check` function for a description of input
    parameters, errors and exceptions.

    Returns
    -------
    is_biased : boolean
        ``True`` if any sub-group pair has significantly (according to the
        specified ``threshold``) different predictive performance, ``False``
        otherwise.
    """
    grid_check = systematic_performance_bias_grid_check(
        metrics_list, threshold)
    is_biased = grid_check.any()
    return is_biased
