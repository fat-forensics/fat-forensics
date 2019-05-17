"""
Tests methods for evaluating model fairness.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
#         Rafael Poyiadzi <rp13102@bristol.ac.uk>
# License: BSD 3 clause

import pytest

import numpy as np

import fatf.fairness.models.measures as ffmm

# ACC: 10/19; TPR: (0) 5/9 (2) 1/3; PPV: (0) 5/7 (2) 1/3
CM1 = np.array([[5, 2, 0], [3, 4, 2], [1, 1, 1]])
# ACC: 6/12; TPR: (0) 3/5 (2) 1/4; PPV: (0) 3/6 (2) 1/2
CM2 = np.array([[3, 0, 3], [2, 2, 0], [0, 1, 1]])
# ACC: 26/32; TPR: (0) 3/3 (2) 20/23; PPV: (0) 3/9 (2) 20/20
CM3 = np.array([[3, 3, 3], [0, 3, 0], [0, 0, 20]])
CM_LIST = [CM1, CM2, CM3]

GROUND_TRUTH = np.array(9 * ['a'] + 5 * ['a'] + 3 * ['a']
                        + 7 * ['b'] + 3 * ['b'] + 6 * ['b']
                        + 3 * ['c'] + 4 * ['c'] + 23 * ['c'])  # yapf: disable
PREDICTIONS = np.array(5 * ['a'] + 3 * ['b'] + 1 * ['c']
                       + 3 * ['a'] + 2 * ['b'] + 0 * ['c']
                       + 3 * ['a'] + 0 * ['b'] + 0 * ['c']
                       #
                       + 2 * ['a'] + 4 * ['b'] + 1 * ['c']
                       + 0 * ['a'] + 2 * ['b'] + 1 * ['c']
                       + 3 * ['a'] + 3 * ['b'] + 0 * ['c']
                       #
                       + 0 * ['a'] + 2 * ['b'] + 1 * ['c']
                       + 3 * ['a'] + 0 * ['b'] + 1 * ['c']
                       + 3 * ['a'] + 0 * ['b'] + 20 * ['c'])  # yapf: disable


def test_validate_tolerance():
    """
    Tests :func:`fatf.fairness.models.measures._validate_tolerance` function.
    """
    value_error = 'The tolerance parameter should be within [0, 1] range.'
    type_error = 'The tolerance parameter should be a number.'

    with pytest.raises(TypeError) as exin:
        ffmm._validate_tolerance('a')
    assert str(exin.value) == type_error

    with pytest.raises(ValueError) as exin:
        ffmm._validate_tolerance(-0.00000001)
    assert str(exin.value) == value_error

    with pytest.raises(ValueError) as exin:
        ffmm._validate_tolerance(1.00000001)
    assert str(exin.value) == value_error

    assert ffmm._validate_tolerance(1.0000000)


def test_equal_accuracy():
    """
    Tests :func:`fatf.fairness.models.measures.equal_accuracy` function.
    """
    ok_array = np.array([[False, False, True], [False, False, True],
                         [True, True, False]])
    not_ok_array = np.array([[False, False, False], [False, False, False],
                             [False, False, False]])

    disparity = ffmm.equal_accuracy(CM_LIST)
    assert np.array_equal(disparity, ok_array)

    disparity = ffmm.equal_accuracy(CM_LIST, tolerance=0.35)
    assert np.array_equal(disparity, not_ok_array)


def test_equal_opportunity():
    """
    Tests :func:`fatf.fairness.models.measures.equal_opportunity` function.
    """
    ok_array = np.array([[False, False, True], [False, False, True],
                         [True, True, False]])
    not_ok_array = np.array([[False, False, True], [False, False, False],
                             [True, False, False]])

    disparity = ffmm.equal_opportunity(CM_LIST)
    assert np.array_equal(disparity, ok_array)

    disparity = ffmm.equal_opportunity(CM_LIST, label_index=2)
    assert np.array_equal(disparity, ok_array)

    disparity = ffmm.equal_opportunity(CM_LIST, tolerance=0.4)
    assert np.array_equal(disparity, not_ok_array)


def test_demographic_parity():
    """
    Tests :func:`fatf.fairness.models.measures.demographic_parity` function.
    """
    ok_array = np.array([[False, True, True], [True, False, False],
                         [True, False, False]])
    not_ok_array = np.array([[False, False, True], [False, False, True],
                             [True, True, False]])

    disparity = ffmm.demographic_parity(CM_LIST)
    assert np.array_equal(disparity, ok_array)

    disparity = ffmm.demographic_parity(CM_LIST, label_index=2)
    assert np.array_equal(disparity, not_ok_array)

    disparity = ffmm.demographic_parity(CM_LIST, label_index=2, tolerance=0.67)
    assert not disparity.any()


def test_disparate_impact_check():
    """
    Tests :func:`fatf.fairness.models.measures.disparate_impact_check`.
    """
    ok_array = np.array([[False, False], [False, False]])
    not_ok_array = np.array([[False, True], [True, False]])

    assert not ffmm.disparate_impact_check(ok_array)
    assert ffmm.disparate_impact_check(not_ok_array)


def test_disparate_impact_grid():
    """
    Tests :func:`fatf.fairness.models.measures._disparate_impact_grid`.
    """
    type_error = ('Criterion has to either be a string indicating parity '
                  'metric or None for the default parity metric (equal '
                  'accuracy).')
    value_error = ('Unrecognised criterion. The following options are '
                   "allowed: ['demographic parity', 'equal opportunity', "
                   "'equal accuracy'].")

    ok_array = np.array([[False, False, True], [False, False, True],
                         [True, True, False]])
    not_ok_array = np.array([[False, True, True], [True, False, False],
                             [True, False, False]])

    with pytest.raises(TypeError) as exin:
        ffmm._disparate_impact_grid(0, 42, 0, 0)
    assert str(exin.value) == type_error

    with pytest.raises(ValueError) as exin:
        ffmm._disparate_impact_grid(0, '42', 0, 0)
    assert str(exin.value) == value_error

    disparity = ffmm._disparate_impact_grid(CM_LIST, None, 0.2, 0)
    assert np.array_equal(disparity, ok_array)

    disparity = ffmm._disparate_impact_grid(CM_LIST, 'equal accuracy', 0.2, 0)
    assert np.array_equal(disparity, ok_array)

    disparity = ffmm._disparate_impact_grid(
        CM_LIST, 'equal opportunity', 0.2, 0)  # yapf: disable
    assert np.array_equal(disparity, ok_array)

    disparity = ffmm._disparate_impact_grid(
        CM_LIST, 'demographic parity', 0.2, 0)  # yapf: disable
    assert np.array_equal(disparity, not_ok_array)


def test_disparate_impact_indexed():
    """
    Tests :func:`fatf.fairness.models.measures.disparate_impact_indexed`.
    """
    grouping = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 17, 18, 19, 20, 21, 22, 23, 33, 34, 35],
        [9, 10, 11, 12, 13, 24, 25, 26, 36, 37, 38, 39],
        [14, 15, 16, 27, 28, 29, 30, 31, 32] + list(range(40, 63))
    ]  # yapf: disable

    disparity = ffmm.disparate_impact_indexed(grouping, GROUND_TRUTH,
                                              PREDICTIONS)
    ok_array = np.array([[False, False, True], [False, False, True],
                         [True, True, False]])
    assert np.array_equal(disparity, ok_array)


def test_disparate_impact():
    """
    Tests :func:`fatf.fairness.models.measures.disparate_impact` function.
    """
    dataset = np.array([
        *(9 * [['a', 'a']]), *(5 * [['a', 'b']]), *(3 * [['a', 'c']]),
        *(7 * [['a', 'a']]), *(3 * [['a', 'b']]), *(6 * [['a', 'c']]),
        *(3 * [['a', 'a']]), *(4 * [['a', 'b']]), *(23 * [['a', 'c']])
    ])

    disparity, bin_names = ffmm.disparate_impact(dataset, GROUND_TRUTH,
                                                 PREDICTIONS, 1)

    assert bin_names == ["('a',)", "('b',)", "('c',)"]
    ok_array = np.array([[False, False, True], [False, False, True],
                         [True, True, False]])
    assert np.array_equal(disparity, ok_array)
