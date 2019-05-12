"""
Tests implementations of models accountability measures.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import pytest

import numpy as np

import fatf.accountability.models.measures as famm


def test_systematic_performance_bias_grid():
    """
    Tests grid-based systematic performance bias check function.

    Tests :func:`fatf.accountability.models.metrics.
    systematic_performance_bias_grid` function.
    """
    type_error_metric_out = 'The metrics_list parameter has to be a list.'
    type_error_metric_in = ('All metric values in the metrics_list should be '
                            'numbers.')
    value_error_metric = 'The metrics list cannot be an empty list.'
    #
    type_error_threshold = 'The threshold parameter has to be a number.'
    value_error_threshold = ('The threshold should be between 0 and 1 '
                             'inclusive.')

    with pytest.raises(TypeError) as exin:
        famm.systematic_performance_bias_grid('a')
    assert str(exin.value) == type_error_metric_out

    with pytest.raises(TypeError) as exin:
        famm.systematic_performance_bias_grid([1, 2, 5.5, 'a', 7.7])
    assert str(exin.value) == type_error_metric_in

    with pytest.raises(ValueError) as exin:
        famm.systematic_performance_bias_grid([])
    assert str(exin.value) == value_error_metric

    with pytest.raises(TypeError) as exin:
        famm.systematic_performance_bias_grid([1, 2, 5.5, 7.7], 'a')
    assert str(exin.value) == type_error_threshold

    with pytest.raises(ValueError) as exin:
        famm.systematic_performance_bias_grid([1, 2, 5.5, 7.7], -1)
    assert str(exin.value) == value_error_threshold

    performances = [12, 11, 10]
    grid_check_true = np.array([[False, False, False],
                                [False, False, False],
                                [False, False, False]])  # yapf: disable
    grid_check = famm.systematic_performance_bias_grid(performances)
    assert np.array_equal(grid_check, grid_check_true)

    performances = [7, 8, 9]
    grid_check_true = np.array([[False, False, True],
                                [False, False, False],
                                [True, False, False]])  # yapf: disable
    grid_check = famm.systematic_performance_bias_grid(performances)
    assert np.array_equal(grid_check, grid_check_true)

    performances = [2, 3, 4]
    grid_check_true = np.array([[False, True, True],
                                [True, False, True],
                                [True, True, False]])  # yapf: disable
    grid_check = famm.systematic_performance_bias_grid(performances)
    assert np.array_equal(grid_check, grid_check_true)


def test_systematic_performance_bias():
    """
    Tests systematic performance bias check function.

    Tests :func:`fatf.accountability.models.metrics.
    systematic_performance_bias` function.
    """
    performances = [12, 11, 10]
    grid_check = famm.systematic_performance_bias(performances)
    assert not grid_check

    performances = [7, 8, 9]
    grid_check = famm.systematic_performance_bias(performances)
    assert grid_check

    performances = [2, 3, 4]
    grid_check = famm.systematic_performance_bias(performances)
    assert grid_check
