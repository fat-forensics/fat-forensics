"""
Functions for testing similarity binary mask function.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
# License: new BSD

import pytest

import numpy as np

import fatf

from fatf.exceptions import IncorrectShapeError

import fatf.utils.data.similarity_binary_mask as fuds


NUMERICAL_NP_ARRAY = np.array([
    [0, 0, 1., 0.],
    [1, 0, 2., 4.],
    [0, 1, 3., 0.],
    [2, 1, 2., 1.],
    [1, 0, 1., 2.],
    [0, 1, 1., 0.]])
NUMERICAL_STRUCT_ARRAY = np.array(
    [(0, 0, 1., 0.),
     (1, 0, 2., 4.),
     (0, 1, 3., 0.),
     (2, 1, 2., 1.),
     (1, 0, 1., 2.),
     (0, 1, 1., 0.)],
    dtype=[('a', 'i'), ('b', 'i'), ('c', 'f'), ('d', 'f')])
CATEGORICAL_NP_ARRAY = np.array([
    ['a', 'b', 'c'],
    ['a', 'f', 'g'],
    ['b', 'c', 'c'],
    ['b', 'f', 'c'],
    ['a', 'f', 'c'],
    ['a', 'b', 'g']])
CATEGORICAL_STRUCT_ARRAY = np.array(
    [('a', 'b', 'c'),
     ('a', 'f', 'g'),
     ('b', 'c', 'c'),
     ('b', 'f', 'c'),
     ('a', 'f', 'c'),
     ('a', 'b', 'g')],
    dtype=[('a', 'U1'), ('b', 'U1'), ('c', 'U1')])
MIXED_ARRAY = np.array(
    [(0, 'a', 0.08, 'a'),
     (0, 'f', 0.03, 'bb'),
     (1, 'c', 0.08, 'aa'),
     (1, 'a', 0.73, 'a'),
     (0, 'c', 0.36, 'b'),
     (1, 'f', 0.08, 'bb')],
    dtype=[('a', 'i'), ('b', 'U1'), ('c', 'f'), ('d', 'U2')])

NUMERICAL_NP_BINARY = np.array([
    [1, 1, 1, 1],
    [0, 1, 0, 0],
    [1, 0, 0, 1],
    [0, 0, 0, 0],
    [0, 1, 1, 0],
    [1, 0, 1, 1]])
NUMERICAL_STRUCT_BINARY = np.array(
    [(1, 1, 1, 1),
     (0, 1, 0, 0),
     (1, 0, 0, 1),
     (0, 0, 0, 0),
     (0, 1, 1, 0),
     (1, 0, 1, 1)],
    dtype=[('a', 'i'), ('b', 'i'), ('c', 'i'), ('d', 'i')])
CATEGORICAL_NP_BINARY = np.array([
    [1, 1, 1],
    [1, 0, 0],
    [0, 0, 1],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 0]])
CATEGORICAL_STRUCT_BINARY = np.array([
    (1, 1, 1),
    (1, 0, 0),
    (0, 0, 1),
    (0, 0, 1),
    (1, 0, 1),
    (1, 1, 0)],
    dtype=[('a', 'i'), ('b', 'i'), ('c', 'i')])
MIXED_BINARY = np.array(
    [(1, 1, 1, 1),
     (1, 0, 0, 0),
     (0, 0, 1, 0),
     (0, 1, 0, 1),
     (1, 0, 0, 0),
     (0, 0, 1, 0)],
    dtype=[('a', 'i'), ('b', 'i'), ('c', 'i'), ('d', 'i')])


def test_validate_input():
    """
    Tests :func:`fatf.utils.data.similarity_binary_mask._validate_input`.
    """
    incorrect_shape_dataset = ('The input dataset must be a '
                                '2-dimensional numpy array.')
    type_error_data = ('The input dataset must be of a base type.')
    incorrect_shape_data_row = ('The data_row must either be a '
                                '1-dimensional numpy array or numpy void '
                                'object for structured rows.')
    type_error_data_row = ('The dtype of the data_row is different to '
                           'the dtype of the dataset provided.')
    incorrect_shape_features = ('The data_row must contain the same '
                                'number of features as the dataset provided.')
    
    with pytest.raises(IncorrectShapeError) as exin:
        fuds._validate_input(NUMERICAL_NP_ARRAY[0], None)
    assert str(exin.value) == incorrect_shape_dataset

    with pytest.raises(IncorrectShapeError) as exin:
        fuds._validate_input(NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY)
    assert str(exin.value) == incorrect_shape_data_row

    with pytest.raises(TypeError) as exin:
        fuds._validate_input(np.array([[None, 0], [0, 1]]), np.array([]))
    assert str(exin.value) == type_error_data

    with pytest.raises(TypeError) as exin:
        fuds._validate_input(NUMERICAL_NP_ARRAY, CATEGORICAL_NP_ARRAY[0])
    assert str(exin.value) == type_error_data_row

    with pytest.raises(IncorrectShapeError) as exin:
        fuds._validate_input(NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY[0][0:1])
    assert str(exin.value) == incorrect_shape_features


def test_similarity_binary_mask():
    """
    Tests :func:`fatf.utils.data.similarity_binary_mask.similarity_binary_mask`.
    """
    binary = fuds.similarity_binary_mask(NUMERICAL_NP_ARRAY,
                                         NUMERICAL_NP_ARRAY[0])
    assert np.array_equal(binary, NUMERICAL_NP_BINARY)

    array = np.array([10.]*4)
    binary = fuds.similarity_binary_mask(NUMERICAL_NP_ARRAY, array)
    assert np.array_equal(binary, np.zeros_like(NUMERICAL_NP_BINARY))

    binary = fuds.similarity_binary_mask(NUMERICAL_STRUCT_ARRAY,
                                         NUMERICAL_STRUCT_ARRAY[0])
    assert np.array_equal(binary, NUMERICAL_STRUCT_BINARY)
    
    array = np.array([(5, 5, 5., 5.)], dtype=[('a', 'i'), ('b', 'i'),
                                              ('c', 'f'), ('d', 'f')])[0]
    binary = fuds.similarity_binary_mask( NUMERICAL_STRUCT_ARRAY, array)
    assert np.array_equal(binary, np.zeros_like(NUMERICAL_STRUCT_BINARY))

    binary = fuds.similarity_binary_mask(CATEGORICAL_NP_ARRAY,
                                         CATEGORICAL_NP_ARRAY[0])
    assert np.array_equal(binary, CATEGORICAL_NP_BINARY)

    array = np.array(['z']*3)
    binary = fuds.similarity_binary_mask(CATEGORICAL_NP_ARRAY, array)
    assert np.array_equal(binary, np.zeros_like(CATEGORICAL_NP_BINARY))

    binary = fuds.similarity_binary_mask(CATEGORICAL_STRUCT_ARRAY,
                                         CATEGORICAL_STRUCT_ARRAY[0])
    assert np.array_equal(binary, CATEGORICAL_STRUCT_BINARY)

    array = np.array([('z', 'z', 'z')], dtype=[('a', 'U1'), ('b', 'U1'),
                                               ('c', 'U1')])[0]
    binary = fuds.similarity_binary_mask(CATEGORICAL_STRUCT_ARRAY, array)
    assert np.array_equal(binary, np.zeros_like(CATEGORICAL_STRUCT_BINARY))

    binary = fuds.similarity_binary_mask(MIXED_ARRAY, MIXED_ARRAY[0])
    assert np.array_equal(binary, MIXED_BINARY)

    array = np.array([(2, 'z', 2., 'z')], dtype=[('a', 'i'), ('b', 'U1'),
                                                 ('c', 'f'), ('d', 'U2')])[0]
    binary = fuds.similarity_binary_mask(MIXED_ARRAY, array)
    assert np.array_equal(binary, np.zeros_like(MIXED_BINARY))
