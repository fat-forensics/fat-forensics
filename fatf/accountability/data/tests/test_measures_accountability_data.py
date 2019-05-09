"""
Tests implementations of data accountability measures.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import pytest

import numpy as np

import fatf.accountability.data.measures as fadm


def test_get_weights():
    """
    Tests :func:`fatf.accountability.data.metrics._get_weights` function.
    """
    weights = fadm._get_weights([[0, 4, 6], [3, 2, 5, 1, 8],
                                 [7, 9, 10, 11, 12, 13, 14]])
    true_weights = np.array([
        0.111, 0.067, 0.067, 0.067, 0.111, 0.067, 0.111, 0.048, 0.067, 0.048,
        0.048, 0.048, 0.048, 0.048, 0.048
    ])
    assert np.allclose(weights, true_weights, atol=1e-3)


def test_validate_threshold():
    """
    Tests :func:`fatf.accountability.data.metrics._validate_threshold`.
    """
    type_error = 'The threshold parameter has to be a number.'
    value_error = 'The threshold should be between 0 and 1 inclusive.'

    with pytest.raises(TypeError) as exin:
        fadm._validate_threshold('a')
    assert str(exin.value) == type_error

    with pytest.raises(ValueError) as exin:
        fadm._validate_threshold(-0.00000001)
    assert str(exin.value) == value_error

    with pytest.raises(ValueError) as exin:
        fadm._validate_threshold(1.00000001)
    assert str(exin.value) == value_error


def test_validate_counts():
    """
    Tests :func:`fatf.accountability.data.metrics._validate_counts` function.
    """
    type_error_out = 'The counts parameter has to be a list of integers.'
    type_error_in = 'Counts have to be integers.'
    value_error = 'Counts cannot be negative integers.'

    with pytest.raises(TypeError) as exin:
        fadm._validate_counts('a')
    assert str(exin.value) == type_error_out

    with pytest.raises(TypeError) as exin:
        fadm._validate_counts([1, 2, 3, 4, 5.0, 6, 7])
    assert str(exin.value) == type_error_in

    with pytest.raises(ValueError) as exin:
        fadm._validate_counts([1, 2, 3, 4, -1, 6, 7])
    assert str(exin.value) == value_error


def test_sampling_bias_grid_check():
    """
    Tests :func:`fatf.accountability.data.metrics.sampling_bias_grid_check`.
    """
    counts = [12, 11, 10]
    grid_check_true = np.array([[False, False, False],
                                [False, False, False],
                                [False, False, False]])  # yapf: disable
    grid_check = fadm.sampling_bias_grid_check(counts)
    assert np.array_equal(grid_check, grid_check_true)

    counts = [7, 8, 9]
    grid_check_true = np.array([[False, False, True],
                                [False, False, False],
                                [True, False, False]])  # yapf: disable
    grid_check = fadm.sampling_bias_grid_check(counts)
    assert np.array_equal(grid_check, grid_check_true)

    counts = [2, 3, 4]
    grid_check_true = np.array([[False, True, True],
                                [True, False, True],
                                [True, True, False]])  # yapf: disable
    grid_check = fadm.sampling_bias_grid_check(counts)
    assert np.array_equal(grid_check, grid_check_true)


def test_sampling_bias_check():
    """
    Tests :func:`fatf.accountability.data.metrics.sampling_bias_check`.
    """
    counts = [12, 11, 10]
    assert not fadm.sampling_bias_check(counts)

    counts = [7, 8, 9]
    assert fadm.sampling_bias_check(counts)


def test_sampling_bias_indexed():
    """
    Tests :func:`fatf.accountability.data.metrics.sampling_bias_indexed`.
    """
    type_error_out = 'The indices_per_bin parameter has to be a list.'
    value_error_empty = 'The indices_per_bin list cannot be empty.'
    value_error_negative_index = ('One of the indices is a negative integer '
                                  '-- all should be non-negative.')
    type_error_nonnumber_index = ('Indices should be integers. *{}* is not an '
                                  'integer.')
    type_error_in = ('One of the elements embedded in the indices_per_bin '
                     'list is not a list.')
    value_error_duplicates = 'Some of the indices are duplicated.'

    with pytest.raises(TypeError) as exin:
        fadm.sampling_bias_indexed('list')
    assert str(exin.value) == type_error_out

    with pytest.raises(TypeError) as exin:
        fadm.sampling_bias_indexed([[1], [2], 'list', [4]])
    assert str(exin.value) == type_error_in

    with pytest.raises(TypeError) as exin:
        fadm.sampling_bias_indexed([[1], [2], [3, 'list'], [4]])
    assert str(exin.value) == type_error_nonnumber_index.format('list')

    with pytest.raises(ValueError) as exin:
        fadm.sampling_bias_indexed([])
    assert str(exin.value) == value_error_empty

    with pytest.raises(ValueError) as exin:
        fadm.sampling_bias_indexed([[1], [2], [-1], [4]])
    assert str(exin.value) == value_error_negative_index

    with pytest.raises(ValueError) as exin:
        fadm.sampling_bias_indexed([[0, 1], [2], [3, 0], [4]])
    assert str(exin.value) == value_error_duplicates

    user_warning = ('The following indices are missing (based on the top '
                    'index): {}.\nIt is possible that more indices are '
                    'missing if they were the last one(s).')
    with pytest.warns(UserWarning) as w:
        counts, weights = fadm.sampling_bias_indexed([[0, 4], [3, 2, 5], [6]])
    assert len(w) == 1
    assert str(w[0].message) == user_warning.format('{1}')
    assert counts == [2, 3, 1]
    true_weights = np.array([0.167, np.nan, 0.111, 0.111, 0.167, 0.111, 0.333])
    assert np.allclose(weights, true_weights, atol=1e-3, equal_nan=True)

    binning = [[0, 4, 6], [3, 2, 5, 1, 8], [7, 9, 10, 11, 12, 13, 14]]
    true_counts = [3, 5, 7]
    true_weights = np.array([
        0.111, 0.067, 0.067, 0.067, 0.111, 0.067, 0.111, 0.048, 0.067, 0.048,
        0.048, 0.048, 0.048, 0.048, 0.048
    ])
    counts, weights = fadm.sampling_bias_indexed(binning)
    assert counts == true_counts
    assert np.allclose(weights, true_weights, atol=1e-3)


def test_sampling_bias():
    """
    Tests :func:`fatf.accountability.data.metrics.sampling_bias` function.
    """
    dataset = np.array([
        ['a', '0', '1'],
        ['b', '1', '2'],
        ['b', '1', '3'],
        ['b', '1', '3'],
        ['b', '0', '3'],
        ['b', '1', '3'],
        ['b', '0', '3'],
        ['b', '2', '3'],
        ['b', '1', '3'],
        ['b', '2', '3'],
        ['b', '2', '3'],
        ['b', '2', '3'],
        ['a', '2', '3'],
        ['b', '2', '2'],
        ['b', '2', '1']
    ])  # yapf: disable
    true_counts = [3, 5, 7]
    true_weights = np.array([
        0.111, 0.067, 0.067, 0.067, 0.111, 0.067, 0.111, 0.048, 0.067, 0.048,
        0.048, 0.048, 0.048, 0.048, 0.048
    ])
    true_bin_names = ["('0',)", "('1',)", "('2',)"]

    counts, weights, bin_names = fadm.sampling_bias(dataset, 1)
    assert counts == true_counts
    assert np.allclose(weights, true_weights, atol=1e-3)
    assert bin_names == true_bin_names
