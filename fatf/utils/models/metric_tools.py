"""
Holds model metric tools.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import inspect

from numbers import Number
from typing import Callable, List, Optional, Tuple, Union
from typing import Dict  # pylint: disable=unused-import

import numpy as np

from fatf.exceptions import IncorrectShapeError

import fatf.utils.array.validation as fuav
import fatf.utils.data.tools as fudt
import fatf.utils.models.metrics as fumm

__all__ = ['confusion_matrix_per_subgroup',
           'confusion_matrix_per_subgroup_indexed',
           'apply_metric_function',
           'apply_metric',
           'performance_per_subgroup',
           'performance_per_subgroup_indexed']  # yapf: disable

Index = Union[int, str]  # A column index type


def confusion_matrix_per_subgroup(
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
    sub-population.

    For warnings raised by this method please see the documentation of
    :func:`fatf.utils.data.tools.validate_indices_per_bin` function.

    Parameters
    ----------
    dataset, column_index, groupings, and numerical_bins_number
        These parameters are described in the documentation of
        :func:`fatf.utils.data.tools.group_by_column` function and are used to
        define a grouping (i.e. sub-populations). If you have your own
        index-based grouping and would like to get sub-population-based
        confusion matrices, please consider using :func:`fatf.utils.models.
        metric_tools.confusion_matrix_per_subgroup_indexed` function.
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
    # pylint: disable=too-many-arguments
    indices_per_bin, bin_names = fudt.group_by_column(
        dataset, column_index, groupings, numerical_bins_number)

    assert fudt.validate_indices_per_bin(indices_per_bin), \
        'Binned indices list is invalid.'

    population_confusion_matrix = confusion_matrix_per_subgroup_indexed(
        indices_per_bin, ground_truth, predictions, labels)
    return population_confusion_matrix, bin_names


def confusion_matrix_per_subgroup_indexed(
        indices_per_bin: List[np.ndarray],
        ground_truth: np.ndarray,
        predictions: np.ndarray,
        labels: Optional[List[Union[str, Number]]] = None) -> List[np.ndarray]:
    """
    Computes confusion matrices for every defined sub-population.

    This is useful for computing a variety of performance metrics based on
    predefined instance index binning for each sub-population.

    This is an alternative to
    :func:`fatf.utils.models.metric_tools.confusion_matrix_per_subgroup`
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
        labels = np.sort(
            np.unique(np.concatenate([ground_truth, predictions]))).tolist()

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
        metrics.append(metric_function(cmx, *args, **kwargs))  # type: ignore

    for metric_value in metrics:
        if not isinstance(metric_value, Number):
            raise TypeError('One of the metric function outputs is not a '
                            'number: *{}*.'.format(metric_value))

    return metrics


def apply_metric(population_confusion_matrix: List[np.ndarray],
                 metric: Optional[str] = None,
                 label_index: int = 0,
                 **kwargs) -> List[Number]:
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
        groupings: Optional[List[Union[Number, Tuple[str]]]] = None,
        numerical_bins_number: int = 5,
        #
        labels: Optional[List[Union[str, Number]]] = None,
        #
        metric: Optional[str] = None,
        metric_function: Optional[Callable[[np.ndarray], Number]] = None,
        #
        **kwargs) -> Tuple[List[Number], List[str]]:
    """
    Computes a chosen metric per sub-population for a data set.

    This function combines
    :func:`fatf.utils.models.metric_tools.confusion_matrix_per_subgroup`
    function together with :func:`fatf.utils.models.metric_tools.apply_metric`
    (when using ``metric`` parameter) and
    :func:`fatf.utils.models.metric_tools.apply_metric_function` (when using
    ``metric_function`` parameter) functions. For the description of
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
    population_cmxs, bin_names = confusion_matrix_per_subgroup(
        dataset, ground_truth, predictions, column_index, groupings,
        numerical_bins_number, labels)

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
        labels: Optional[List[Union[str, Number]]] = None,
        #
        metric: Optional[str] = None,
        metric_function: Optional[Callable[[np.ndarray], Number]] = None,
        #
        **kwargs) -> List[Number]:
    """
    Computes a chosen metric per sub-population for index-based grouping.

    This function combines :func:`fatf.utils.models.metric_tools.
    confusion_matrix_per_subgroup_indexed` function together with
    :func:`fatf.utils.models.metric_tools.apply_metric` (when using ``metric``
    parameter) and :func:`fatf.utils.models.metric_tools.apply_metric_function`
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
    population_cmxs = confusion_matrix_per_subgroup_indexed(
        indices_per_bin, ground_truth, predictions, labels)

    if metric_function is not None:
        population_metrics = apply_metric_function(
            population_cmxs, metric_function, *args, **kwargs)
    else:
        population_metrics = apply_metric(population_cmxs, metric, label_index,
                                          **kwargs)

    return population_metrics
