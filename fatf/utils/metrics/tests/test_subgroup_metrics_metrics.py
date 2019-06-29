"""
Tests implementations of sub-population model metrics and tools.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import pytest

import numpy as np

import fatf.utils.metrics.subgroup_metrics as fums

MISSING_LABEL_WARNING = ('Some of the given labels are not present in either '
                         'of the input arrays: {2}.')

DATASET = np.array([['0', '3', '0'], ['0', '5', '0'], ['0', '7', '0'],
                    ['0', '5', '0'], ['0', '7', '0'], ['0', '3', '0'],
                    ['0', '5', '0'], ['0', '3', '0'], ['0', '7', '0'],
                    ['0', '5', '0'], ['0', '7', '0'], ['0', '7', '0'],
                    ['0', '5', '0'], ['0', '7', '0'], ['0', '7', '0']])
_INDICES_PER_BIN = [[0, 5, 7], [1, 6, 9, 3, 12], [2, 4, 8, 10, 11, 13, 14]]
GROUND_TRUTH = np.zeros((15, ), dtype=int)
GROUND_TRUTH[_INDICES_PER_BIN[0]] = [0, 1, 0]
GROUND_TRUTH[_INDICES_PER_BIN[1]] = [0, 1, 2, 1, 0]
GROUND_TRUTH[_INDICES_PER_BIN[2]] = [0, 1, 2, 0, 1, 2, 0]
PREDICTIONS = np.zeros((15, ), dtype=int)
PREDICTIONS[_INDICES_PER_BIN[0]] = [0, 0, 0]
PREDICTIONS[_INDICES_PER_BIN[1]] = [0, 2, 2, 2, 0]
PREDICTIONS[_INDICES_PER_BIN[2]] = [0, 1, 2, 2, 1, 0, 0]


def test_apply_metric_function():
    """
    Tests :func:`fatf.utils.metrics.subgroup_metrics.apply_metric_function`.
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
        return 'zero'  # pragma: nocover

    def one(one):
        return 0.5

    def one_array(one):
        return one.sum()

    def two(one, two):
        return 'one' + '+' + 'two' + ' : ' + two

    cfmx = np.array([[1, 2], [2, 1]])

    with pytest.raises(TypeError) as exin:
        fums.apply_metric_function('a', None)
    assert str(exin.value) == type_error_cmxs

    with pytest.raises(ValueError) as exin:
        fums.apply_metric_function([], None)
    assert str(exin.value) == value_error_cmxs

    with pytest.raises(TypeError) as exin:
        fums.apply_metric_function([cfmx], None)
    assert str(exin.value) == type_error_fn

    with pytest.raises(AttributeError) as exin:
        fums.apply_metric_function([cfmx], zero)
    assert str(exin.value) == attribute_error_fn

    with pytest.raises(TypeError) as exin:
        fums.apply_metric_function([cfmx], two, 'second_arg')
    assert str(exin.value) == type_error_metric.format('one+two : second_arg')

    measures = fums.apply_metric_function([cfmx], one)
    assert measures == [0.5]

    measures = fums.apply_metric_function([cfmx], one_array)
    assert measures == [6]


def test_apply_metric():
    """
    Tests :func:`fatf.utils.metrics.subgroup_metrics.apply_metric` function.
    """
    type_error = 'The metric parameter has to be a string.'
    value_error = ('The selected metric (*{}*) is not recognised. The '
                   'following options are available: {}.')

    available_metrics = [
        'true positive rate', 'true negative rate', 'false positive rate',
        'false negative rate', 'positive predictive value', 'accuracy',
        'treatment', 'negative predictive value'
    ]

    cfmx = np.array([[1, 2], [3, 4]])

    with pytest.raises(TypeError) as exin:
        fums.apply_metric([cfmx], 5)
    assert str(exin.value) == type_error

    with pytest.raises(ValueError) as exin:
        fums.apply_metric([cfmx], 'unknown_metric')
    assert str(exin.value) == value_error.format('unknown_metric',
                                                 sorted(available_metrics))

    measures = fums.apply_metric([cfmx])
    assert len(measures) == 1
    assert measures[0] == 0.5

    measures = fums.apply_metric([cfmx], 'true positive rate')
    assert len(measures) == 1
    assert measures[0] == 0.25

    measures = fums.apply_metric([cfmx], 'true positive rate', label_index=1)
    assert len(measures) == 1
    assert measures[0] == pytest.approx(0.667, abs=1e-3)


def test_performance_per_subgroup():
    """
    Tests :func:`fatf.utils.metrics.subgroup_metrics.performance_per_subgroup`.
    """
    true_bin_names = ["('3',)", "('5',)", "('7',)"]

    # Default metric
    with pytest.warns(UserWarning) as w:
        bin_metrics, bin_names = fums.performance_per_subgroup(
            DATASET, GROUND_TRUTH, PREDICTIONS, 1)
    assert len(w) == 1
    assert str(w[0].message) == MISSING_LABEL_WARNING
    #
    assert bin_metrics == pytest.approx([2 / 3, 3 / 5, 5 / 7], abs=1e-3)
    assert bin_names == true_bin_names

    # Named metric
    with pytest.warns(UserWarning) as w:
        bin_metrics, bin_names = fums.performance_per_subgroup(
            DATASET, GROUND_TRUTH, PREDICTIONS, 1, metric='true positive rate')
    assert len(w) == 1
    assert str(w[0].message) == MISSING_LABEL_WARNING
    #
    assert bin_metrics == pytest.approx([1, 1, 2 / 3], abs=1e-3)
    assert bin_names == true_bin_names

    # Named metric with **kwargs
    with pytest.warns(UserWarning) as w:
        bin_metrics, bin_names = fums.performance_per_subgroup(
            DATASET,
            GROUND_TRUTH,
            PREDICTIONS,
            1,
            metric='true negative rate',
            strict=True)
    assert len(w) == 1
    assert str(w[0].message) == MISSING_LABEL_WARNING
    #
    assert bin_metrics == pytest.approx([0, 1 / 3, 3 / 4], abs=1e-3)
    assert bin_names == true_bin_names

    def one(one):
        return one.sum()

    def two(one, two, three=0):
        return one.sum() + two + three

    # Function metric -- takes the precedence
    with pytest.warns(UserWarning) as w:
        bin_metrics, bin_names = fums.performance_per_subgroup(
            DATASET,
            GROUND_TRUTH,
            PREDICTIONS,
            1,
            metric='true negative rate',
            metric_function=one)
    assert len(w) == 1
    assert str(w[0].message) == MISSING_LABEL_WARNING
    #
    assert bin_metrics == [3, 5, 7]
    assert bin_names == true_bin_names

    # Function metric with *args
    with pytest.warns(UserWarning) as w:
        bin_metrics, bin_names = fums.performance_per_subgroup(
            DATASET,
            GROUND_TRUTH,
            PREDICTIONS,
            1,
            3,
            metric='true negative rate',
            metric_function=two)
    assert len(w) == 1
    assert str(w[0].message) == MISSING_LABEL_WARNING
    #
    assert bin_metrics == [6, 8, 10]
    assert bin_names == true_bin_names

    # Function metric with *args and **kwargs
    with pytest.warns(UserWarning) as w:
        bin_metrics, bin_names = fums.performance_per_subgroup(
            DATASET,
            GROUND_TRUTH,
            PREDICTIONS,
            1,
            3,
            metric='true negative rate',
            metric_function=two,
            three=-6)
    assert len(w) == 1
    assert str(w[0].message) == MISSING_LABEL_WARNING
    #
    assert bin_metrics == [0, 2, 4]
    assert bin_names == true_bin_names


def test_performance_per_subgroup_indexed():
    """
    Tests calculating performance per indexed sub-group.

    Tests :func:`fatf.utils.metrics.subgroup_metrics.
    performance_per_subgroup_indexed` function.
    """
    # Default metric
    with pytest.warns(UserWarning) as w:
        bin_metrics = fums.performance_per_subgroup_indexed(
            _INDICES_PER_BIN, GROUND_TRUTH, PREDICTIONS, labels=[0, 1, 2])
    assert len(w) == 1
    assert str(w[0].message) == MISSING_LABEL_WARNING
    #
    assert bin_metrics == pytest.approx([2 / 3, 3 / 5, 5 / 7], abs=1e-3)

    # Named metric
    with pytest.warns(UserWarning) as w:
        bin_metrics = fums.performance_per_subgroup_indexed(
            _INDICES_PER_BIN,
            GROUND_TRUTH,
            PREDICTIONS,
            labels=[0, 1, 2],
            metric='true positive rate')
    assert len(w) == 1
    assert str(w[0].message) == MISSING_LABEL_WARNING
    #
    assert bin_metrics == pytest.approx([1, 1, 2 / 3], abs=1e-3)

    # Named metric with **kwargs
    with pytest.warns(UserWarning) as w:
        bin_metrics = fums.performance_per_subgroup_indexed(
            _INDICES_PER_BIN,
            GROUND_TRUTH,
            PREDICTIONS,
            labels=[0, 1, 2],
            metric='true negative rate',
            strict=True)
    assert len(w) == 1
    assert str(w[0].message) == MISSING_LABEL_WARNING
    #
    assert bin_metrics == pytest.approx([0, 1 / 3, 3 / 4], abs=1e-3)

    def one(one):
        return one.sum()

    def two(one, two, three=0):
        return one.sum() + two + three

    # Function metric -- takes the precedence
    with pytest.warns(UserWarning) as w:
        bin_metrics = fums.performance_per_subgroup_indexed(
            _INDICES_PER_BIN,
            GROUND_TRUTH,
            PREDICTIONS,
            labels=[0, 1, 2],
            metric='true negative rate',
            metric_function=one)
    assert len(w) == 1
    assert str(w[0].message) == MISSING_LABEL_WARNING
    #
    assert bin_metrics == [3, 5, 7]

    # Function metric with *args
    with pytest.warns(UserWarning) as w:
        bin_metrics = fums.performance_per_subgroup_indexed(
            _INDICES_PER_BIN,
            GROUND_TRUTH,
            PREDICTIONS,
            3,
            labels=[0, 1, 2],
            metric='true negative rate',
            metric_function=two)
    assert len(w) == 1
    assert str(w[0].message) == MISSING_LABEL_WARNING
    #
    assert bin_metrics == [6, 8, 10]

    # Function metric with *args and **kwargs
    with pytest.warns(UserWarning) as w:
        bin_metrics = fums.performance_per_subgroup_indexed(
            _INDICES_PER_BIN,
            GROUND_TRUTH,
            PREDICTIONS,
            3,
            labels=[0, 1, 2],
            metric='true negative rate',
            metric_function=two,
            three=-6)
    assert len(w) == 1
    assert str(w[0].message) == MISSING_LABEL_WARNING
    #
    assert bin_metrics == [0, 2, 4]
