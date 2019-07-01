"""
The :mod:`fatf.accountability.models.measures` module implements accountability
measures for models.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
#         Rafael Poyiadzi <rp13102@bristol.ac.uk>
# License: new BSD

from numbers import Number
from typing import List

import numpy as np

__all__ = ['systematic_performance_bias', 'systematic_performance_bias_grid']


def systematic_performance_bias(metrics_list: List[float],
                                threshold: float = 0.8) -> bool:
    """
    Checks for a systematic bias in provided predictive performance values.

    Please see the documentation of :func:`fatf.accountability.models.\
measures.systematic_performance_bias_grid` function for a description of input
    parameters, errors and exceptions.

    .. note::
       This function expects a list of predictive performance per sub-group for
       tested data. To get this list please use either of the following
       functions:
       :func:`fatf.utils.metrics.subgroup_metrics.performance_per_subgroup`/
       :func:`fatf.utils.metrics.subgroup_metrics.\
performance_per_subgroup_indexed` or
       :func:`fatf.utils.metrics.tools.confusion_matrix_per_subgroup`/
       :func:`fatf.utils.metrics.tools.confusion_matrix_per_subgroup_indexed`
       in conjunction with
       :func:`fatf.utils.metrics.subgroup_metrics.apply_metric_function`/
       :func:`fatf.utils.metrics.subgroup_metrics.apply_metric`.

    Returns
    -------
    is_biased : boolean
        ``True`` if any sub-group pair has significantly (according to the
        specified ``threshold``) different predictive performance, ``False``
        otherwise.
    """
    grid_check = systematic_performance_bias_grid(metrics_list, threshold)
    is_biased = grid_check.any()
    return is_biased


def systematic_performance_bias_grid(metrics_list: List[float],
                                     threshold: float = 0.8) -> np.ndarray:
    """
    Checks for pairwise systematic bias in group-wise predictive performance.

    If a disparity in performance is found to be above the specified
    ``threshold`` a given pair sub-population performance metrics is considered
    biased.

    .. note::
       This function expects a list of predictive performance per sub-group for
       tested data. To get this list please use either of the following
       functions:
       :func:`fatf.utils.metrics.subgroup_metrics.performance_per_subgroup`/
       :func:`fatf.utils.metrics.subgroup_metrics.\
performance_per_subgroup_indexed` or
       :func:`fatf.utils.metrics.tools.confusion_matrix_per_subgroup`/
       :func:`fatf.utils.metrics.tools.confusion_matrix_per_subgroup_indexed`
       in conjunction with
       :func:`fatf.utils.metrics.subgroup_metrics.apply_metric_function`/
       :func:`fatf.utils.metrics.subgroup_metrics.apply_metric`.

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
    np.logical_or(grid_check, grid_check.T, out=grid_check)

    return grid_check
