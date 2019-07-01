"""
The :mod:`fatf.utils.metrics.subgroup_metrics` module holds sub-group metrics.

These functions are mainly used to compute a given performance metric for every
sub population in a data set defined by a grouping on a selected feature.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import inspect

from numbers import Number
from typing import Callable, List, Optional, Tuple, Union
from typing import Dict  # pylint: disable=unused-import

import numpy as np

import fatf.utils.metrics.metrics as fumm
import fatf.utils.metrics.tools as fumt

__all__ = ['apply_metric_function',
           'apply_metric',
           'performance_per_subgroup',
           'performance_per_subgroup_indexed']  # yapf: disable

Index = Union[int, str]  # A column index type


def apply_metric_function(population_confusion_matrix: List[np.ndarray],
                          metric_function: Callable[[np.ndarray], float],
                          *args, **kwargs) -> List[float]:
    """
    Applies the provided performance metric to every confusion matrix.

    The performance metric function needs to take a numpy.ndarray confusion
    matrix as the first parameter followed by any number of unnamed and named
    parameters provided by ``*args`` and ``**kwargs`` parameters.

    Parameters
    ----------
    population_confusion_matrix : List[numpy.ndarray]
        A list of confusion matrices for each sub-population.
    metric_function : Callable[[numpy.ndarray], Number]
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
            assert fumt.validate_confusion_matrix(confusion_matrix), \
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
        metrics.append(metric_function(cmx, *args, **kwargs))  # type: ignore

    for metric_value in metrics:
        if not isinstance(metric_value, Number):
            raise TypeError('One of the metric function outputs is not a '
                            'number: *{}*.'.format(metric_value))

    return metrics


def apply_metric(population_confusion_matrix: List[np.ndarray],
                 metric: Optional[str] = None,
                 label_index: int = 0,
                 **kwargs) -> List[float]:
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
    }  # type: Dict[str, Callable]

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
                                        available_metrics[metric], **kwargs)
    else:
        metrics = apply_metric_function(population_confusion_matrix,
                                        available_metrics[metric], label_index,
                                        **kwargs)

    return metrics


def performance_per_subgroup(
        dataset: np.ndarray,
        #
        ground_truth: np.ndarray,
        predictions: np.ndarray,
        #
        column_index: Index,
        #
        *args,
        label_index: int = 0,
        #
        groupings: Optional[List[Union[float, Tuple[str]]]] = None,
        numerical_bins_number: int = 5,
        treat_as_categorical: Optional[bool] = None,
        #
        labels: Optional[List[Union[str, float]]] = None,
        #
        metric: Optional[str] = None,
        metric_function: Optional[Callable[[np.ndarray], float]] = None,
        #
        **kwargs) -> Tuple[List[float], List[str]]:
    """
    Computes a chosen metric per sub-population for a data set.

    This function combines
    :func:`fatf.utils.metrics.tools.confusion_matrix_per_subgroup`
    function together with
    :func:`fatf.utils.metrics.subgroup_metrics.apply_metric` (when using
    ``metric`` parameter) and
    :func:`fatf.utils.metrics.subgroup_metrics.apply_metric_function` (when
    using ``metric_function`` parameter) functions. For the description of
    parameters, errors and exceptions please see the documentation of these
    functions.

    .. note::
       The ``metric_function`` parameter takes the precedence over the
       ``metric`` parameter is both are provided.

    Returns
    -------
    population_metrics : List[numbers]
        A list with the value of the selected metric for every sub-population.
    bin_names : List[strings]
        The name of every sub-population (binning results) defined by the
        feature ranges for a numerical feature and feature value sets for a
        categorical feature.
    """
    # pylint: disable=too-many-locals
    population_cmxs, bin_names = fumt.confusion_matrix_per_subgroup(
        dataset, ground_truth, predictions, column_index, groupings,
        numerical_bins_number, treat_as_categorical, labels)

    if metric_function is not None:
        population_metrics = apply_metric_function(
            population_cmxs, metric_function, *args, **kwargs)
    else:
        population_metrics = apply_metric(population_cmxs, metric, label_index,
                                          **kwargs)

    return population_metrics, bin_names


def performance_per_subgroup_indexed(
        indices_per_bin: List[np.ndarray],
        ground_truth: np.ndarray,
        predictions: np.ndarray,
        #
        *args,
        label_index: int = 0,
        #
        labels: Optional[List[Union[str, float]]] = None,
        #
        metric: Optional[str] = None,
        metric_function: Optional[Callable[[np.ndarray], float]] = None,
        #
        **kwargs) -> List[float]:
    """
    Computes a chosen metric per sub-population for index-based grouping.

    This function combines
    :func:`fatf.utils.metrics.tools.confusion_matrix_per_subgroup_indexed`
    function together with
    :func:`fatf.utils.metrics.subgroup_metrics.apply_metric` (when using
    ``metric`` parameter) and
    :func:`fatf.utils.metrics.subgroup_metrics.apply_metric_function`
    (when using ``metric_function`` parameter) functions. For the description
    of parameters, errors and exceptions please see the documentation of these
    functions.

    .. note::
       The ``metric_function`` parameter takes the precedence over the
       ``metric`` parameter is both are provided.

    Returns
    -------
    population_metrics : List[numbers]
        A list with the value of the selected metric for every sub-population.
    bin_names : List[strings]
        The name of every sub-population (binning results) defined by the
        feature ranges for a numerical feature and feature value sets for a
        categorical feature.
    """
    population_cmxs = fumt.confusion_matrix_per_subgroup_indexed(
        indices_per_bin, ground_truth, predictions, labels)

    if metric_function is not None:
        population_metrics = apply_metric_function(
            population_cmxs, metric_function, *args, **kwargs)
    else:
        population_metrics = apply_metric(population_cmxs, metric, label_index,
                                          **kwargs)

    return population_metrics
