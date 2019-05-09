"""
Tests implementations of models accountability measures.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import pytest

import numpy as np

from fatf.exceptions import IncorrectShapeError

import fatf.accountability.models.measures as famm


def test_systematic_performance_bias_grid_check():
    """
    Tests grid-based systematic performance bias check function.

    Tests :func:`fatf.accountability.models.metrics.
    systematic_performance_bias_grid_check` function.
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
        famm.systematic_performance_bias_grid_check('a')
    assert str(exin.value) == type_error_metric_out

    with pytest.raises(TypeError) as exin:
        famm.systematic_performance_bias_grid_check([1, 2, 5.5, 'a', 7.7])
    assert str(exin.value) == type_error_metric_in

    with pytest.raises(ValueError) as exin:
        famm.systematic_performance_bias_grid_check([])
    assert str(exin.value) == value_error_metric

    with pytest.raises(TypeError) as exin:
        famm.systematic_performance_bias_grid_check([1, 2, 5.5, 7.7], 'a')
    assert str(exin.value) == type_error_threshold

    with pytest.raises(ValueError) as exin:
        famm.systematic_performance_bias_grid_check([1, 2, 5.5, 7.7], -1)
    assert str(exin.value) == value_error_threshold

    performances = [12, 11, 10]
    grid_check_true = np.array([[False, False, False],
                                [False, False, False],
                                [False, False, False]])
    grid_check = famm.systematic_performance_bias_grid_check(performances)
    assert np.array_equal(grid_check, grid_check_true)

    performances = [7, 8, 9]
    grid_check_true = np.array([[False, False, True],
                                [False, False, False],
                                [True, False, False]])
    grid_check = famm.systematic_performance_bias_grid_check(performances)
    assert np.array_equal(grid_check, grid_check_true)

    performances = [2, 3, 4]
    grid_check_true = np.array([[False, True, True],
                                [True, False, True],
                                [True, True, False]])
    grid_check = famm.systematic_performance_bias_grid_check(performances)
    assert np.array_equal(grid_check, grid_check_true)


def test_systematic_performance_bias_check():
    """
    Tests systematic performance bias check function.

    Tests :func:`fatf.accountability.models.metrics.
    systematic_performance_bias_check` function.
    """
    performances = [12, 11, 10]
    grid_check = famm.systematic_performance_bias_check(performances)
    assert not grid_check

    performances = [7, 8, 9]
    grid_check = famm.systematic_performance_bias_check(performances)
    assert grid_check

    performances = [2, 3, 4]
    grid_check = famm.systematic_performance_bias_check(performances)
    assert grid_check


def test_apply_metric_function():
    """
    Tests :func:`fatf.accountability.models.metrics.apply_metric_function`.
    """
    type_error_cmxs = ('The population_confusion_matrix parameter has to be a '
                       'list.')
    value_error_cmxs = ('The population_confusion_matrix parameter cannot be '
                        'an empty list.')
    #
    type_error_fn = ('The metric_function parameter has to be a Python '
                     'callable.')
    attribute_error_fn = ('The metric_function callable needs to have at '
                          'least one required parameter taking a confusion '
                          'matrix. 0 were found.')
    #
    type_error_metric = ('One of the metric function outputs is not a number: '
                         '*{}*.')

    def zero():
        return 'zero'

    def one(one):
        return 0.5

    def one_array(one):
        return one.sum()

    def two(one, two):
        return 'one' + '+' + 'two' + ' : ' + two

    cfmx = np.array([[1, 2], [2, 1]])

    with pytest.raises(TypeError) as exin:
        famm.apply_metric_function('a', None)
    assert str(exin.value) == type_error_cmxs

    with pytest.raises(ValueError) as exin:
        famm.apply_metric_function([], None)
    assert str(exin.value) == value_error_cmxs

    with pytest.raises(TypeError) as exin:
        famm.apply_metric_function([cfmx], None)
    assert str(exin.value) == type_error_fn

    with pytest.raises(AttributeError) as exin:
        famm.apply_metric_function([cfmx], zero)
    assert str(exin.value) == attribute_error_fn

    with pytest.raises(TypeError) as exin:
        famm.apply_metric_function([cfmx], two, 'second_arg')
    assert str(exin.value) == type_error_metric.format('one+two : second_arg')

    measures = famm.apply_metric_function([cfmx], one)
    assert measures == [0.5]

    measures = famm.apply_metric_function([cfmx], one_array)
    assert measures == [6]


def test_apply_metric():
    """
    Tests :func:`fatf.accountability.models.metrics.apply_metric` function.
    """
    type_error = 'The metric parameter has to be a string.'
    value_error = ('The selected metric (*{}*) is not recognised. The '
                   'following options are available: {}.')

    available_metrics = ['true positive rate', 'true negative rate',
                         'false positive rate', 'false negative rate',
                         'positive predictive value', 'accuracy', 'treatment',
                         'negative predictive value']

    cfmx = np.array([[1, 2], [3, 4]])

    with pytest.raises(TypeError) as exin:
        famm.apply_metric([cfmx], 5)
    assert str(exin.value) == type_error

    with pytest.raises(ValueError) as exin:
        famm.apply_metric([cfmx], 'unknown_metric')
    assert str(exin.value) == value_error.format('unknown_metric',
                                                 sorted(available_metrics))

    measures = famm.apply_metric([cfmx])
    assert len(measures) == 1
    assert measures[0] == 0.5

    measures = famm.apply_metric([cfmx], 'true positive rate')
    assert len(measures) == 1
    assert measures[0] == 0.25

    measures = famm.apply_metric([cfmx], 'true positive rate', label_index=1)
    assert len(measures) == 1
    assert measures[0] == pytest.approx(0.667, abs=1e-3)


def test_systematic_performance_bias_indexed():
    """
    Tests indexed systematic performance bias evaluation.

    Tests :func:`fatf.accountability.models.metrics.
    systematic_performance_bias_indexed` function.
    """
    incorrect_shape_error_gt = ('The ground_truth parameter should be a '
                                '1-dimensional numpy array.')
    incorrect_shape_error_p = ('The predictions parameter should be a '
                               '1-dimensional numpy array.')

    flat = np.array([1,2])
    square = np.array([[1, 2], [3, 4]])
    with pytest.raises(IncorrectShapeError) as exin:
        famm.systematic_performance_bias_indexed([[0]], square, square)
    assert str(exin.value) == incorrect_shape_error_gt

    with pytest.raises(IncorrectShapeError) as exin:
        famm.systematic_performance_bias_indexed([[0]], flat, square)
    assert str(exin.value) == incorrect_shape_error_p

    indices_per_bin = [[0, 5, 7], [1, 6, 9, 3, 12], [2, 4, 8, 10, 11, 13, 14]]
    ground_truth = np.zeros((15, ), dtype=int)
    ground_truth[indices_per_bin[0]] = [0, 1, 0]
    ground_truth[indices_per_bin[1]] = [0, 1, 2, 1, 0]
    ground_truth[indices_per_bin[2]] = [0, 1, 2, 0, 1, 2, 0]
    predictions = np.zeros((15, ), dtype=int)
    predictions[indices_per_bin[0]] = [0, 0, 0]
    predictions[indices_per_bin[1]] = [0, 2, 2, 2, 0]
    predictions[indices_per_bin[2]] = [0, 1, 2, 2, 1, 0, 0]

    mx1 = np.array([[2, 1, 0], [0, 0, 0], [0, 0, 0]])
    mx2 = np.array([[2, 0, 0], [0, 0, 0], [0, 2, 1]])
    mx3 = np.array([[2, 0, 1], [0, 2, 0], [1, 0, 1]])

    with pytest.warns(UserWarning) as w:
        pcmxs_1 = famm.systematic_performance_bias_indexed(
            indices_per_bin, ground_truth, predictions, labels=[0, 1, 2])
        pcmxs_2 = famm.systematic_performance_bias_indexed(
            indices_per_bin, ground_truth, predictions)
    assert len(w) == 2
    wmsg = ('Some of the given labels are not present in either of the input '
            'arrays: {2}.')
    assert str(w[0].message) == wmsg
    assert str(w[1].message) == wmsg
    assert len(pcmxs_1) == 3
    assert len(pcmxs_2) == 3
    assert np.array_equal(pcmxs_1[0], mx1)
    assert np.array_equal(pcmxs_2[0], mx1)
    assert np.array_equal(pcmxs_1[1], mx2)
    assert np.array_equal(pcmxs_2[1], mx2)
    assert np.array_equal(pcmxs_1[2], mx3)
    assert np.array_equal(pcmxs_2[2], mx3)


def test_systematic_performance_bias():
    """
    Tests systematic performance bias evaluation.

    Tests
    :func:`fatf.accountability.models.metrics.systematic_performance_bias`
    function.
    """
    dataset = np.array([
        ['0', '3', '0'],
        ['0', '5', '0'],
        ['0', '7', '0'],
        ['0', '5', '0'],
        ['0', '7', '0'],
        ['0', '3', '0'],
        ['0', '5', '0'],
        ['0', '3', '0'],
        ['0', '7', '0'],
        ['0', '5', '0'],
        ['0', '7', '0'],
        ['0', '7', '0'],
        ['0', '5', '0'],
        ['0', '7', '0'],
        ['0', '7', '0']])
    indices_per_bin = [[0, 5, 7], [1, 6, 9, 3, 12], [2, 4, 8, 10, 11, 13, 14]]
    ground_truth = np.zeros((15, ), dtype=int)
    ground_truth[indices_per_bin[0]] = [0, 1, 0]
    ground_truth[indices_per_bin[1]] = [0, 1, 2, 1, 0]
    ground_truth[indices_per_bin[2]] = [0, 1, 2, 0, 1, 2, 0]
    predictions = np.zeros((15, ), dtype=int)
    predictions[indices_per_bin[0]] = [0, 0, 0]
    predictions[indices_per_bin[1]] = [0, 2, 2, 2, 0]
    predictions[indices_per_bin[2]] = [0, 1, 2, 2, 1, 0, 0]

    mx1 = np.array([[2, 1, 0], [0, 0, 0], [0, 0, 0]])
    mx2 = np.array([[2, 0, 0], [0, 0, 0], [0, 2, 1]])
    mx3 = np.array([[2, 0, 1], [0, 2, 0], [1, 0, 1]])

    with pytest.warns(UserWarning) as w:
        pcmxs, bin_names = famm.systematic_performance_bias(
            dataset, ground_truth, predictions, 1)
    assert len(pcmxs) == 3
    assert len(w) == 1
    assert str(w[0].message) == ('Some of the given labels are not present in '
                                 'either of the input arrays: {2}.')

    assert np.array_equal(pcmxs[0], mx1)
    assert np.array_equal(pcmxs[1], mx2)
    assert np.array_equal(pcmxs[2], mx3)
    assert bin_names == ["('3',)", "('5',)", "('7',)"]
