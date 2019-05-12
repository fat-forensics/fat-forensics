"""
Tests implementations of data fairness measures.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import pytest

import numpy as np

from fatf.exceptions import IncorrectShapeError

import fatf.fairness.data.measures as ffdm


def test_systemic_bias():
    """
    Tests :func:`fatf.fairness.data.measures.systemic_bias` function.
    """
    incorrect_shape_data = 'The dataset should be a 2-dimensional numpy array.'
    incorrect_shape_gt = ('The ground truth should be a 1-dimensional numpy '
                          'array.')
    incorrect_shape_length = ('The number of rows in the dataset and the '
                              'ground truth should be equal.')
    index_error = ('The following protected feature indices are not valid for '
                   'the dataset array: {}.')
    type_error = 'The protected_features parameter should be a list.'
    value_error = 'Some of the protected indices are duplicated.'

    one_d_array = np.array([1, 2])
    two_d_array = np.array([[1, 2, 3], [4, 5, 6]])

    with pytest.raises(IncorrectShapeError) as exin:
        ffdm.systemic_bias(one_d_array, two_d_array, None)
    assert str(exin.value) == incorrect_shape_data

    with pytest.raises(IncorrectShapeError) as exin:
        ffdm.systemic_bias(two_d_array, two_d_array, None)
    assert str(exin.value) == incorrect_shape_gt

    with pytest.raises(IncorrectShapeError) as exin:
        ffdm.systemic_bias(two_d_array, np.array([1, 2, 3]), None)
    assert str(exin.value) == incorrect_shape_length

    with pytest.raises(TypeError) as exin:
        ffdm.systemic_bias(two_d_array, one_d_array, None)
    assert str(exin.value) == type_error

    with pytest.raises(IndexError) as exin:
        ffdm.systemic_bias(two_d_array, one_d_array, ['z', 'a'])
    assert str(exin.value) == index_error.format("['a', 'z']")

    with pytest.raises(ValueError) as exin:
        ffdm.systemic_bias(two_d_array, one_d_array, [2, 1, 2])
    assert str(exin.value) == value_error

    data = np.array([[5, 2, 3],
                     [5, 4, 3],
                     [3, 2, 3],
                     [5, 4, 2],
                     [3, 2, 3]])  # yapf: disable
    data_struct = np.array(
        [(5, '02', 3),
         (5, '04', 3),
         (3, '02', 3),
         (5, '04', 2),
         (3, '02', 3)],
        dtype=[('a', int), ('b', 'U2'), ('c', float)])  # yapf: disable
    ground_truth = np.array(['a', 'a', 'b', 'b', 'a'])

    zero_indices = []
    one_indices = [2]
    two_indices = [1, 2]
    all_indices = [0, 1, 2]
    #
    one_indices_struct = ['c']
    two_indices_struct = ['b', 'c']
    all_indices_struct = ['a', 'b', 'c']

    zero_matrix = np.array([[False, False, False, False, False],
                            [False, False, False, False, False],
                            [False, False, False, False, True],
                            [False, False, False, False, False],
                            [False, False, True, False, False]])
    one_matrix = np.array([[False, False, False, False, False],
                           [False, False, False, True, False],
                           [False, False, False, False, True],
                           [False, True, False, False, False],
                           [False, False, True, False, False]])
    two_matrix = np.array([[False, False, False, True, False],
                           [False, False, False, True, False],
                           [False, False, False, False, True],
                           [True, True, False, False, False],
                           [False, False, True, False, False]])
    all_matrix = np.array([[False, False, True, True, False],
                           [False, False, True, True, False],
                           [True, True, False, False, True],
                           [True, True, False, False, True],
                           [False, False, True, True, False]])

    # Test unstructured data -- zero feature
    grid = ffdm.systemic_bias(data, ground_truth, zero_indices)
    assert np.array_equal(grid, zero_matrix)
    # Test unstructured data -- one feature
    grid = ffdm.systemic_bias(data, ground_truth, one_indices)
    assert np.array_equal(grid, one_matrix)
    # Test unstructured data -- two features
    grid = ffdm.systemic_bias(data, ground_truth, two_indices)
    assert np.array_equal(grid, two_matrix)
    # Test unstructured data -- all features
    grid = ffdm.systemic_bias(data, ground_truth, all_indices)
    assert np.array_equal(grid, all_matrix)

    # Test structured data -- zero feature
    grid = ffdm.systemic_bias(data_struct, ground_truth, zero_indices)
    assert np.array_equal(grid, zero_matrix)
    # Test structured data -- one feature
    grid = ffdm.systemic_bias(data_struct, ground_truth, one_indices_struct)
    assert np.array_equal(grid, one_matrix)
    # Test structured data -- two features
    grid = ffdm.systemic_bias(data_struct, ground_truth, two_indices_struct)
    assert np.array_equal(grid, two_matrix)
    # Test structured data -- all features
    grid = ffdm.systemic_bias(data_struct, ground_truth, all_indices_struct)
    assert np.array_equal(grid, all_matrix)


def test_systemic_bias_check():
    """
    Tests :func:`fatf.fairness.data.measures.systemic_bias_check` function.
    """
    incorrect_shape_error = 'The systemic bias matrix has to be 2-dimensional.'
    incorrect_shape_error_square = 'The systemic bias matrix has to be square.'
    type_error = 'The systemic bias matrix has to be of boolean type.'
    value_error_symmetric = ('The systemic bias matrix has to be diagonally '
                             'symmetric.')
    value_error_structured = ('The systemic bias matrix cannot be a '
                              'structured numpy array.')

    with pytest.raises(IncorrectShapeError) as exin:
        ffdm.systemic_bias_check(np.array([1, 2, 3]))
    assert str(exin.value) == incorrect_shape_error

    with pytest.raises(IncorrectShapeError) as exin:
        ffdm.systemic_bias_check(np.array([[True], [True]]))
    assert str(exin.value) == incorrect_shape_error_square

    with pytest.raises(ValueError) as exin:
        ffdm.systemic_bias_check(np.array([(True, )], dtype=[('a', bool)]))
    assert str(exin.value) == value_error_structured

    with pytest.raises(TypeError) as exin:
        ffdm.systemic_bias_check(np.array([[1, 2], [3, 4]]))
    assert str(exin.value) == type_error

    with pytest.raises(ValueError) as exin:
        ffdm.systemic_bias_check(np.array([[True, False], [False, False]]))
    assert str(exin.value) == value_error_symmetric

    with pytest.raises(ValueError) as exin:
        ffdm.systemic_bias_check(np.array([[True, False], [True, True]]))
    assert str(exin.value) == value_error_symmetric

    ok_array = np.array([[False, True, False], [True, False, False],
                         [False, False, False]])
    assert ffdm.systemic_bias_check(ok_array)

    not_ok_array = np.array([[False, False, False], [False, False, False],
                             [False, False, False]])
    assert not ffdm.systemic_bias_check(not_ok_array)
