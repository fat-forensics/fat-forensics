"""
This module tests feature choice functions
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
# License: new BSD

import pytest

import numpy as np

import fatf
import fatf.transparency.sklearn.feature_choice as ftsfc
from fatf.exceptions import IncorrectShapeError

# yapf: disable
ONE_D_ARRAY = np.array([0, 4, 3, 0])
NUMERICAL_NP_ARRAY_TARGET = np.array([2, 0, 1, 1, 0, 2])
NUMERICAL_NP_ARRAY = np.array([
    [0, 0, 0.08, 0.69],
    [1, 0, 0.03, 0.29],
    [0, 1, 0.99, 0.82],
    [2, 1, 0.73, 0.48],
    [1, 0, 0.36, 0.89],
    [0, 1, 0.07, 0.21]])
NUMERICAL_STRUCT_ARRAY = np.array(
    [(0, 0, 0.08, 0.69),
     (1, 0, 0.03, 0.29),
     (0, 1, 0.99, 0.82),
     (2, 1, 0.73, 0.48),
     (1, 0, 0.36, 0.89),
     (0, 1, 0.07, 0.21)],
    dtype=[('a', 'i'), ('b', 'i'), ('c', 'f'), ('d', 'f')])
CATEGORICAL_NP_ARRAY = np.array([
    ['a', 'b', 'c'],
    ['a', 'f', 'g'],
    ['b', 'c', 'c'],
    ['b', 'f', 'c'],
    ['a', 'f', 'c'],
    ['a', 'b', 'g']])
# yapf:enable


def test_is_input_valid():
    """
    Tests :func:`fatf.transparency.sklearn.feature_choice._is_input_valid`.
    """
    dataset_shape_msg = 'The input dataset must be a 2-dimensional array.'
    dataset_type_msg = 'The input dataset must only contain numerical dtypes'
    target_shape_msg = 'The input target array must a 1-dimensional array.'
    target_shape_2_msg = ('The number of labels in target must be the same as '
                          'the number of samples in dataset.')
    weights_shape_msg = 'The input weights array must a 1-dimensional array.'
    weights_shape_2_msg = ('The number distances in weights must be the same '
                           'as the number of samples in dataset.')
    num_features_type_msg = 'num_features must be an integer.'
    num_features_value_msg = ('num_features must be an integer greater than '
                              'zero.')

    with pytest.raises(IncorrectShapeError) as exin:
        ftsfc._is_input_valid(ONE_D_ARRAY, None, None, None)
    assert str(exin.value) == dataset_shape_msg

    with pytest.raises(TypeError) as exin:
        ftsfc._is_input_valid(np.array([[None, 0], [0, 1]]), None, None, None)
    assert str(exin.value) == dataset_type_msg

    with pytest.raises(TypeError) as exin:
       ftsfc._is_input_valid(CATEGORICAL_NP_ARRAY, None, None, None)
    assert str(exin.value) == dataset_type_msg

    with pytest.raises(IncorrectShapeError) as exin:
        ftsfc._is_input_valid(NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY, None,
                             None)
    assert str(exin.value) == target_shape_msg

    with pytest.raises(IncorrectShapeError) as exin:
        ftsfc._is_input_valid(
            NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY_TARGET[0:4], None, None)

    assert str(exin.value) == target_shape_2_msg

    with pytest.raises(IncorrectShapeError) as exin:
        ftsfc._is_input_valid(NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY_TARGET,
                              NUMERICAL_NP_ARRAY, None)
    assert str(exin.value) == weights_shape_msg

    with pytest.raises(IncorrectShapeError) as exin:
        ftsfc._is_input_valid(NUMERICAL_STRUCT_ARRAY, NUMERICAL_NP_ARRAY_TARGET,
                             NUMERICAL_NP_ARRAY_TARGET[0:4], None)
    assert str(exin.value) == weights_shape_2_msg

    with pytest.raises(TypeError) as exin:
        ftsfc._is_input_valid(NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY_TARGET,
                              NUMERICAL_NP_ARRAY_TARGET, 'a')
    assert str(exin.value) == num_features_type_msg

    with pytest.raises(ValueError) as exin:
        ftsfc._is_input_valid(NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY_TARGET,
                              NUMERICAL_NP_ARRAY_TARGET, 0)
    assert str(exin.value) == num_features_value_msg

    # All good
    ftsfc._is_input_valid(NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY_TARGET,
                          NUMERICAL_NP_ARRAY_TARGET, 2)
    ftsfc._is_input_valid(NUMERICAL_STRUCT_ARRAY, NUMERICAL_NP_ARRAY_TARGET,
                          NUMERICAL_NP_ARRAY_TARGET, 3)


def test_lasso_path():
    """
    Tests :func:`fatf.transparency.sklearn.feature_choice.lasso_path`.
    """
    lasso_path_msg = ('Not enough nonzero coefficients were found by '
                      '\'sklearn.linear_model.lars_path\' function. This '
                      'could be due to the weights being too small.')
    fatf.setup_random_seed()

    weights = np.ones((NUMERICAL_NP_ARRAY.shape[0], ))
    features = ftsfc.lasso_path(
        NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY_TARGET, weights, 2)
    assert np.array_equal(features, np.array([0, 1]))
    features = ftsfc.lasso_path(
        NUMERICAL_STRUCT_ARRAY, NUMERICAL_NP_ARRAY_TARGET, weights, 2)
    assert np.array_equal(features, np.array(['a', 'b']))

    weights = np.array([1, 1, 100, 1, 1, 1])
    features = ftsfc.lasso_path(
        NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY_TARGET, weights, 2)
    assert np.array_equal(features, np.array([0, 2]))
    features = ftsfc.lasso_path(
        NUMERICAL_STRUCT_ARRAY, NUMERICAL_NP_ARRAY_TARGET, weights, 2)
    assert np.array_equal(features, np.array(['a', 'c']))

    weights = np.array([1, 1, 100, 1, 1, 1]) * 1e-20
    with pytest.raises(ValueError) as exin:
        ftsfc.lasso_path(NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY_TARGET,
                         weights, 2)
    assert str(exin.value) == lasso_path_msg
