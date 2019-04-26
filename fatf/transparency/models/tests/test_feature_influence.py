"""
Functions for testing ICE and PD calculations.

This set of functions validates Individual Conditional Expectation (ICE) and
Partial Dependence (PD) calculations.

"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
#         Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import pytest

import numpy as np

import fatf
import fatf.transparency.models.feature_influence as ftmfi
import fatf.utils.models as fum

from fatf.exceptions import IncompatibleModelError, IncorrectShapeError
from fatf.utils.testing.arrays import (BASE_NP_ARRAY, BASE_STRUCTURED_ARRAY,
                                       NOT_BASE_NP_ARRAY)

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
    ['b', 'c', 'c']])
CATEGORICAL_STRUCT_ARRAY = np.array(
    [('a', 'b', 'c'),
     ('a', 'f', 'g'),
     ('b', 'c', 'c')],
    dtype=[('a', 'U1'), ('b', 'U1'), ('c', 'U1')])
MIXED_ARRAY = np.array(
    [(0, 'a', 0.08, 'a'),
     (0, 'f', 0.03, 'bb'),
     (1, 'c', 0.99, 'aa'),
     (1, 'a', 0.73, 'a'),
     (0, 'c', 0.36, 'b'),
     (1, 'f', 0.07, 'bb')],
    dtype=[('a', 'i'), ('b', 'U1'), ('c', 'f'), ('d', 'U2')])

NUMERICAL_NP_ARRAY_TEST_INT = np.array([
    [1, 0, 0, 0],
    [0, 0, 0, 0]])
NUMERICAL_STRUCT_ARRAY_TEST_INT = np.array([
    [1, 0, 0, 0],
    [0, 0, 0, 0]],
    dtype=[('a', 'i'), ('b', 'i'), ('c', 'i'), ('d', 'i')])
NUMERICAL_NP_ARRAY_TEST = np.array([
    [1, 0, 0.03, 0.5],
    [0, 0, 0.56, 0.32]])
NUMERICAL_STRUCT_ARRAY_TEST = np.array(
    [(1, 0, 0.03, 0.5),
     (0, 0, 0.56, 0.32)],
    dtype=[('a', 'i'), ('b', 'i'), ('c', 'f'), ('d', 'f')])
NUMERICAL_NP_ICE = np.array([
    [[1., 0., 0.],
     [1., 0., 0.],
     [1., 0., 0.]],
    [[0.0, 0., 1.0],
     [0.5, 0., 0.5],
     [0.5, 0., 0.5]]])
NUMERICAL_NP_ICE_2D = np.array([
    [[[0.5, 0.0, 0.5],
      [0.5, 0.0, 0.5],
      [0.5, 0.0, 0.5]],
     [[0.5, 0.0, 0.5],
      [0.5, 0.0, 0.5],
      [0.5, 0.0, 0.5]],
     [[1.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [1.0, 0.0, 0.0]]],
    [[[0.0, 0.0, 1.0],
      [0.5, 0.0, 0.5],
      [0.5, 0.0, 0.5]],
     [[1.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [0.5, 0.0, 0.5]],
     [[1.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [1.0, 0.0, 0.0]]]])
NUMERICAL_NP_ICE_REGRESSION = np.array([
    [[0.],
    [0.],
    [0.]],
    [[2.],
    [1.],
    [1.]]])
NUMERICAL_NP_PD = np.array([
    [0.50, 0.0, 0.50],
    [0.75, 0.0, 0.25],
    [0.75, 0.0, 0.25]])
NUMERICAL_NP_PD_2D = np.array([
   [[0.25, 0.0, 0.75],
    [0.50, 0.0, 0.50],
    [0.50, 0.0, 0.50]],
   [[0.75, 0.0, 0.25],
    [0.75, 0.0, 0.25],
    [0.50, 0.0, 0.50]],
   [[1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0]]])
NUMERICAL_NP_PD_REGRESSION = np.array([
    [1.],
    [0.5],
    [0.5]])
NUMERICAL_NP_ICE_CAT = np.array([
    [[1., 0., 0.],
     [1., 0., 0.]],
    [[0.0, 0., 1.0],
     [0.5, 0., 0.5]]])
NUMERICAL_NP_ICE_CAT_2D = np.array([
    [[[0.5, 0., 0.5],
      [0.5, 0., 0.5]],
     [[0.5, 0., 0.5],
      [0.5, 0., 0.5]],
     [[1., 0., 0.],
      [1., 0., 0.]]],
    [[[0., 0., 1.],
      [0.5, 0., 0.5]],
     [[1., 0., 0.],
      [0.5, 0., 0.5]],
     [[1., 0., 0.],
      [1., 0., 0.]]]])
NUMERICAL_NP_PD_CAT_2D = np.array([
   [[0.25, 0.0, 0.75],
    [0.50, 0.0, 0.50]],
   [[0.75, 0.0, 0.25],
    [0.50, 0.0, 0.50]],
   [[1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0]]])
NUMERICAL_NP_ICE_CAT_REGRESSION = np.array([
    [[0.],
    [0.]],
    [[2.],
    [1.]]])
NUMERICAL_NP_PD_CAT = np.array([
    [0.50, 0.0, 0.50],
    [0.75, 0.0, 0.25]])
NUMERICAL_NP_PD_CAT_REGRESSION = np.array([
    [1.],
    [0.5]])
NUMERICAL_NP_ICE_100 = np.array(
    [100 * [[1.0, 0.0, 0.0]],
     46 * [[0.0, 0.0, 1.0]] + 54 * [[0.5, 0.0, 0.5]]])
NUMERICAL_NP_ICE_100_2D = np.array(
   [2*[100 * [[0.5, 0.0, 0.5]]] + 
   1*[100 * [[1.0, 0.0, 0.0]]]] +
   [1*[46 * [[0.0, 0.0, 1.0]] + 54 * [[0.5, 0.0, 0.5]]] + 
   1*[59 * [[1.0, 0.0, 0.0]] + 41 * [[0.5, 0.0, 0.5]]]+ 
   1*[100 * [[1.0, 0.0, 0.0]]]])
NUMERICAL_NP_PD_100_2D = np.array(
    [46*[[0.25, 0.0, 0.75]] + 54*[[0.5, 0.0, 0.5]]] + 
    [59*[[0.75, 0.0, 0.25]] + 41*[[0.5, 0.0, 0.5]]] + 
    [100*[[1.0, 0.0, 0.0]]])
"""
NUMERICAL_NP_ICE_100_2D = np.array(
   [2*[100 * [[0.5, 0.0, 0.5]]] + 1*[100 * [[1.0, 0.0, 0.0]]]]
   +[1*[46*[[0.0, 0.0, 1.0] + 54 * [[0.5, 0.0, 0.5]]]]
   + 1*[46*[[0.0, 0.0, 1.0] + 54 * [[0.5, 0.0, 0.5]]]]
   1*[100 * [[1.0, 0.0, 0.0]]]])
"""
NUMERICAL_NP_ICE_100_REGRESSION = np.array(
    [100 * [[0.0]],
     46 * [[2.0]] + 54 * [[1.0]]])
NUMERICAL_NP_PD_100 = np.array(
    46 * [[0.5, 0.0, 0.5]] + 54 * [[0.75, 0.00, 0.25]])
NUMERICAL_NP_PD_100_REGRESSION = np.array(
    46 * [[1.0]] + 54 * [[0.5]])
NUMERICAL_NP_LINESPACE = np.array([0.32, 0.41, 0.5])
NUMERICAL_NP_LINESPACE_2D = (np.array([0., 0.5, 1.]), 
                             np.array([0.32, 0.41, 0.5]))
NUMERICAL_NP_LINESPACE_CAT = np.array([0.32, 0.5])
NUMERICAL_NP_LINESPACE_CAT_2D = (np.array([0., 0.5, 1.]), 
                                 np.array([0.32, 0.5]))
NUMERICAL_NP_LINESPACE_100 = np.linspace(0.32, 0.5, 100)
NUMERICAL_NP_LINESPACE_100_2D = (np.array([0., 0.5, 1.]),
                                 np.linspace(0.32, 0.5, 100))
NUMERICAL_NP_VARIANCE = np.array([
    [0.25, 0., 0.25],
    [0.0625, 0., 0.0625],
    [0.0625, 0., 0.0625]])
NUMERICAL_NP_VARIANCE_REGRESSION = np.array([
    [1.0],
    [0.25],
    [0.25]])
NUMERICAL_NP_VARIANCE_CAT = np.array([
    [0.25, 0., 0.25],
    [0.0625, 0., 0.0625]])
NUMERICAL_NP_VARIANCE_CAT_REGRESSION = np.array([
    [1.0],
    [0.25]])
NUMERICAL_NP_VARIANCE_100 = np.array(
    46 * [[0.25, 0., 0.25]] + 54 * [[0.0625, 0., 0.0625]])
NUMERICAL_NP_VARIANCE_100_REGRESSION = np.array(
    46 * [[1.0]] + 54 * [[0.25]])
CATEGORICAL_NP_ARRAY_TEST = np.array([
    ['a', 'f', 'g'],
    ['b', 'f', 'c']])
CATEGORICAL_STRUCT_ARRAY_TEST = np.array(
    [('a', 'f', 'g'),
     ('b', 'f', 'c')],
    dtype=[('a', 'U1'), ('b', 'U1'), ('c', 'U1')])
CATEGORICAL_NP_ARRAY_TARGET = np.array([0, 1, 1])
CATEGORICAL_NP_ICE = np.array([
    [[0.5, 0.5],
     [0.5, 0.5]],
    [[0.0, 1.0],
     [0.0, 1.0]]])
CATEGORICAL_NP_ICE_2D = np.array([
    [[[0.5, 0.5],
      [0.5, 0.5]],
     [[0.0, 1.0],
      [0.0, 1.0]]],
    [[[0.5, 0.5],
      [0.5, 0.5]],
     [[0.0, 1.0],
      [0.0, 1.0]]]])
CATEGORICAL_NP_ICE_REGRESSION = np.array([
    [[0.5,],
     [0.5]],
    [[1.0],
     [1.0]]])
CATEGORICAL_NP_PD = np.array([
    [0.25, 0.75],
    [0.25, 0.75]])
CATEGORICAL_NP_PD_2D = np.array([
    [[0.50, 0.50],
     [0.50, 0.50]],
    [[0.0, 1.0],
     [0.0, 1.0]]])
CATEGORICAL_NP_PD_REGRESSION = np.array([
    [0.75],
    [0.75]])
CATEGORICAL_NP_LINESPACE = np.array(['c', 'g'])
CATEGORICAL_NP_LINESPACE_2D = (np.array(['a', 'b']), np.array(['c', 'g']))
CATEGORICAL_NP_VARIANCE = np.array([
    [0.0625, 0.0625],
    [0.0625, 0.0625]])
CATEGORICAL_NP_VARIANCE_REGRESSION = np.array([
    [0.0625],
    [0.0625]])
MIXED_ARRAY_TEST = np.array(
    [(0, 'a', 0.08, 'a'),
     (1, 'a', 0.88, 'bb'),
     (1, 'f', 0.07, 'bb')],
    dtype=[('a', 'i'), ('b', 'U1'), ('c', 'f'), ('d', 'U2')])
MIXED_ARRAY_TARGET = np.array(['a', 'b', 'c', 'a', 'b', 'c'])
MIXED_ARRAY_TARGET_REGRESSION = np.array([0, 1, 2, 0, 1, 2])
MIXED_ICE_NUMERICAL = np.array([
    [[1.0, 0.0, 0.0],
     [1.0, 0.0, 0.0],
     [1.0, 0.0, 0.0]],
    [[0.0, 0.5, 0.5],
     [0.0, 0.5, 0.5],
     [0.0, 0.5, 0.5]]])
MIXED_ICE_NUMERICAL_2D = np.array([
    [[[1.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [1.0, 0.0, 0.0]],
     [[1.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [1.0, 0.0, 0.0]],
     [[1.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [1.0, 0.0, 0.0]]],
    [[[0.0, 0.5, 0.5],
      [0.0, 0.5, 0.5],
      [0.0, 0.5, 0.5]],
     [[0.0, 0.5, 0.5],
      [0.0, 0.5, 0.5],
      [0.0, 0.5, 0.5]],
     [[0.0, 0.5, 0.5],
      [0.0, 0.5, 0.5],
      [0.0, 0.5, 0.5]]]])
MIXED_ICE_NUMERICAL_REGRESSION = np.array([
    [[0.0],
     [0.0],
     [0.0]],
    [[1.5],
     [1.5],
     [1.5]]])
MIXED_PD_NUMERICAL = np.array([
    [0.5, 0.25, 0.25],
    [0.5, 0.25, 0.25],
    [0.5, 0.25, 0.25]])
MIXED_PD_NUMERICAL_2D = np.array([
    [[0.50, 0.25, 0.25],
     [0.50, 0.25, 0.25],
     [0.50, 0.25, 0.25]],
    [[0.50, 0.25, 0.25],
     [0.50, 0.25, 0.25],
     [0.50, 0.25, 0.25]],
    [[0.50, 0.25, 0.25],
     [0.50, 0.25, 0.25],
     [0.50, 0.25, 0.25]]])
MIXED_PD_NUMERICAL_REGRESSION = np.array([
    [0.75],
    [0.75],
    [0.75]])
MIXED_LINESPACE_NUMERICAL = np.array([0, 0.5, 1])
MIXED_LINESPACE_NUMERICAL_2D = (np.array([0.0, 0.5, 1.0]),
                                np.array([0.07, 0.075, 0.08]))
MIXED_VARIANCE_NUMERICAL = np.array([
    [0.25, 0.0625, 0.0625],
    [0.25, 0.0625, 0.0625],
    [0.25, 0.0625, 0.0625]])
MIXED_VARIANCE_NUMERICAL_REGRESSION = np.array([
    [0.5625],
    [0.5625],
    [0.5625]])
MIXED_ICE_CATEGORICAL = np.array([
    [[1.0, 0.0, 0.0],
     [0.5, 0.5, 0.0]],
    [[0.5, 0.0, 0.5],
     [0.0, 0.5, 0.5]]])
MIXED_ICE_CATEGORICAL_2D = np.array([
    [[[1.0, 0.0, 0.0],
      [0.5, 0.5, 0.0]],
     [[0.5, 0.5, 0.0],
      [0.0, 0.5, 0.5]]],
    [[[1.0, 0.0, 0.0],
      [0.5, 0.0, 0.5]],
     [[0.5, 0.0, 0.5],
      [0.0, 0.5, 0.5]]],
    [[[1.0, 0.0, 0.0],
      [0.5, 0.0, 0.5]],
     [[0.5, 0.0, 0.5],
      [0.0, 0.5, 0.5]]]])
MIXED_ICE_CATEGORICAL_REGRESSION = np.array([
    [[0.0],
     [0.5]],
    [[1.0],
     [1.5]]])
MIXED_PD_CATEGORICAL = np.array([
    [0.75, 0.0, 0.25],
    [0.25, 0.5, 0.25]])
MIXED_PD_CATEGORICAL_2D = np.array([
    [[1.00, 0.00, 0.00],
     [0.50, 0.16, 0.33]],
    [[0.50, 0.16, 0.33],
     [0.00, 0.50, 0.50]]])
MIXED_PD_CATEGORICAL_REGRESSION = np.array([
    [0.5],
    [1.0]])
MIXED_LINESPACE_CATEGORICAL = np.array(['a', 'f'])
MIXED_LINESPACE_CATEGORICAL_2D = (np.array(['a', 'f']), np.array(['a', 'bb']))
MIXED_LINESPACE_MIX_2D = (np.array([0.0, 0.5, 1.0]), np.array(['a', 'f']))
MIXED_VARIANCE_CATEGORICAL = np.array([
    [0.0625, 0., 0.0625],
    [0.0625, 0., 0.0625]])
MIXED_VARIANCE_CATEGORICAL_REGRESSION = np.array([
    [0.25],
    [0.25]])
MIXED_ICE_MIX_2D = np.array([
    [[[1.0, 0.0, 0.0],
      [0.5, 0.5, 0.0]],
     [[1.0, 0.0, 0.0],
      [0.5, 0.0, 0.5]],
     [[1.0, 0.0, 0.0],
      [0.5, 0.0, 0.5]]],
    [[[0.5, 0.5, 0.0],
      [0.0, 0.5, 0.5]],
     [[1.0, 0.0, 0.0],
      [0.0, 0.5, 0.5]],
     [[0.5, 0.0, 0.5],
      [0.0, 0.5, 0.5]]],
    [[[0.5, 0.5, 0.0],
      [0.0, 0.5, 0.5]],
     [[0.5, 0.0, 0.5],
      [0.0, 0.5, 0.5]],
     [[0.5, 0.0, 0.5],
      [0.0, 0.5, 0.5]]]])
MIXED_PD_MIX_2D = np.array([
    [[0.66, 0.33, 0.00],
     [0.16, 0.50, 0.33]],
    [[0.83, 0.00, 0.16],
     [0.16, 0.33, 0.50]],
    [[0.66, 0.00, 0.33],
     [0.16, 0.33, 0.50]]])
# yapf: enable


class InvalidModel(object):
    """
    Tests for exceptions when a model lacks the ``predict_proba`` method.
    """

    def __init__(self):
        """
        Initialises not-a-model.
        """
        pass

    def fit(self, X, y):
        """
        Fits not-a-model.
        """
        return X, y  # pragma: nocover

    def predict(self, X):
        """
        Predicts not-a-model.
        """
        return X  # pragma: nocover


class InvalidModelRegression(object):
    """
    Tests for exceptions when model kacs the ``predict`` method.
    """

    def __init__(self):
        """
        Initialises not-a-model.
        """
        pass

    def fit(self, X, y):
        """
        Fits not-a-model.
        """
        return X, y # pragma: nocover


def test_is_valid_input():
    """
    Tests :func:`fatf.transparency.models.feature_influence._is_valid_input`.
    """
    knn_model = fum.KNN()

    # Data
    msg = 'The input dataset must be a 2-dimensional array.'
    with pytest.raises(IncorrectShapeError) as exin:
        ftmfi._input_is_valid(ONE_D_ARRAY, None, None, None)
    assert str(exin.value) == msg

    msg = ('The input dataset must only contain base types (textual and '
           'numerical).')
    with pytest.raises(ValueError) as exin:
        ftmfi._input_is_valid(NOT_BASE_NP_ARRAY, None, None, None)
    assert str(exin.value) == msg

    # Feature index
    msg = 'Provided feature index is not valid for the input dataset.'
    with pytest.raises(IndexError) as exin:
        ftmfi._input_is_valid(BASE_STRUCTURED_ARRAY, 0, None, None)
    assert str(exin.value) == msg

    msg = 'Provided feature index is not valid for the input dataset.'
    with pytest.raises(IndexError) as exin:
        ftmfi._input_is_valid(BASE_STRUCTURED_ARRAY, [0, 10], None, None)
    assert str(exin.value) == msg

    with pytest.raises(IndexError) as exin:
        ftmfi._input_is_valid(BASE_NP_ARRAY, 'numerical', None,
                              None)
    assert str(exin.value) == msg

    msg = ('feature_index has to be a single value or a list of length two.')
    with pytest.raises(ValueError) as exin:
        ftmfi._input_is_valid(BASE_STRUCTURED_ARRAY, [0, 1, 2], None, None)
    assert str(exin.value) == msg

    # Steps number
    msg = ('steps_number parameter has to either be None, an integer or a list '
          'of None and integers.')
    with pytest.raises(TypeError) as exin:
        ftmfi._input_is_valid(BASE_NP_ARRAY, 1, None, 'a')
    assert str(exin.value) == msg

    msg = ('steps_number parameter has to either be None, an integer or a list '
          'of None and integers.')
    with pytest.raises(TypeError) as exin:
        ftmfi._input_is_valid(BASE_NP_ARRAY, [0, 1], None, [3, 'a'])
    assert str(exin.value) == msg

    msg = 'steps_number has to be at least 2.'
    with pytest.raises(ValueError) as exin:
        ftmfi._input_is_valid(BASE_NP_ARRAY, 1, None, 1)
    assert str(exin.value) == msg

    # Treat as categorical
    msg = ('treat_as_categorical has to either be None, a boolean or a list '
           'of None and booleans.')
    with pytest.raises(TypeError) as exin:
        ftmfi._input_is_valid(BASE_NP_ARRAY, 1, 'a', None)
    assert str(exin.value) == msg

    msg = ('treat_as_categorical has to either be None, a boolean or a list '
           'of None and booleans.')
    with pytest.raises(TypeError) as exin:
        ftmfi._input_is_valid(BASE_NP_ARRAY, [0, 1], [True, 'a'], None)
    assert str(exin.value) == msg

    # List of variables longer than length two or less than 1
    msg = 'steps_number has to be a single value or a list of length two.'
    with pytest.raises(ValueError) as exin:
        ftmfi._input_is_valid(BASE_NP_ARRAY, 0, False, [5, 5, 5])
    assert str(exin.value) == msg

    msg = ('treat_as_categorical has to be a single value or a list of length '
          'two.')
    with pytest.raises(ValueError) as exin:
        ftmfi._input_is_valid(BASE_NP_ARRAY, 0, [True, True, True], 5)
    assert str(exin.value) == msg

    msg = 'steps_number has to be a single value or a list of length two.'
    with pytest.raises(ValueError) as exin:
        ftmfi._input_is_valid(BASE_NP_ARRAY, 0, False, [])
    assert str(exin.value) == msg

    msg = ('treat_as_categorical has to be a single value or a list of length '
          'two.')
    with pytest.raises(ValueError) as exin:
        ftmfi._input_is_valid(BASE_NP_ARRAY, 0, [], 5)
    assert str(exin.value) == msg

    # 1 Feature Index given but 2 options given.
    msg = ('{} feature indices given but {} treat_as_categorical values given. '
           'If one feature index is given, treat_as_categorical must only be '
           'one value.')
    with pytest.raises(ValueError) as exin:
        ftmfi._input_is_valid(BASE_NP_ARRAY, 0, [True, True], 5)
    assert str(exin.value) == msg.format(1, 2)

    msg = ('{} feature indices given but {} steps_number values given. If one '
           'feature index is given, steps_number must only be one value.')
    with pytest.raises(ValueError) as exin:
        ftmfi._input_is_valid(BASE_NP_ARRAY, 0, True, [5, 5])
    assert str(exin.value) == msg.format(1, 2)

    # Functional
    assert ftmfi._input_is_valid(BASE_NP_ARRAY, 1, None, 2)
    assert ftmfi._input_is_valid(BASE_NP_ARRAY, 1, False, 5)
    # Steps number will be ignored anyway
    assert ftmfi._input_is_valid(BASE_NP_ARRAY, 1, True, 2)
    # With list of two
    assert ftmfi._input_is_valid(BASE_NP_ARRAY, [0, 1], False, 3)
    assert ftmfi._input_is_valid(BASE_NP_ARRAY, [0, 1], [False, True], 3)
    assert ftmfi._input_is_valid(BASE_NP_ARRAY, [0, 1], None, 3)
    assert ftmfi._input_is_valid(BASE_NP_ARRAY, [0, 1], False, [3, 3])


def test_generalise_dataset_type():
    """
    Tests dataset generalisation

    This function tests :func:`fatf.transparency.models.feature_influence.
    _generalise_dataset_type`.
    """
    # Test for type generalisation int -> float for classic arrays
    generalised_dataset = ftmfi._generalise_dataset_type(
        NUMERICAL_NP_ARRAY_TEST_INT, 0, np.array([0, 0.5, 1.]), False)

    assert generalised_dataset.dtype == np.float64
    assert np.array_equal(generalised_dataset,
                          NUMERICAL_NP_ARRAY_TEST_INT.astype(np.float64))
    
    # Test for type generalisation int -> float for structured arrays
    new_dtypes = [('a', '<f8'), ('b', 'i'), ('c', 'i'), ('d', 'i')]
    generalised_dataset = ftmfi._generalise_dataset_type(
        NUMERICAL_STRUCT_ARRAY_TEST_INT, 'a', np.array([0, 0.5, 1.]), True)
    assert generalised_dataset.dtype == new_dtypes
    assert np.array_equal(generalised_dataset, 
                          NUMERICAL_STRUCT_ARRAY_TEST_INT.astype(new_dtypes))
    
    # Test for type generalisation int -> float for mixed arrays
    new_dtypes = [('a', '<f8'), ('b', 'U1'), ('c', 'f'), ('d', 'U2')]
    generalised_dataset = ftmfi._generalise_dataset_type(
        MIXED_ARRAY_TEST, 'a', np.array([0, 0.5, 1.]), True)
    assert generalised_dataset.dtype == new_dtypes
    assert np.array_equal(generalised_dataset,
                          MIXED_ARRAY_TEST.astype(new_dtypes))
    
    # Test for type generalised <U1 -> <U2 for classic arrays
    generalised_dataset = ftmfi._generalise_dataset_type(
        CATEGORICAL_NP_ARRAY_TEST, 'a', np.array(['ab', 'cd', 'ef']), False)
    assert generalised_dataset.dtype == '<U2'
    assert np.array_equal(generalised_dataset,
                          CATEGORICAL_NP_ARRAY_TEST.astype('<U2'))
    
     # Test for type generalised <U1 -> <U2 for structured arrays
    new_dtypes = [('a', '<U2'), ('b', 'U1'), ('c', 'U1')]
    generalised_dataset = ftmfi._generalise_dataset_type(
        CATEGORICAL_STRUCT_ARRAY_TEST, 'a', np.array(['ab', 'cd', 'ef']), True)
    assert generalised_dataset.dtype == new_dtypes
    assert np.array_equal(generalised_dataset,
                          CATEGORICAL_STRUCT_ARRAY_TEST.astype(new_dtypes))


def test_get_feature_range():
    """
    Tests feature range calculation.

    This function tests
    :func:`fatf.transparency.models.feature_influence._get_feature_range`.
    """
    # Numerical interpolations with classic array
    values = np.array([0., 1., 2.])
    steps = 3
    interpolated_values, steps_number = ftmfi._get_feature_range(
        NUMERICAL_NP_ARRAY, 0, False, 3, False)
    assert np.array_equal(interpolated_values, values)
    assert steps_number == steps

    # Numerical interpolations with structured array
    interpolated_values, steps_number = ftmfi._get_feature_range(
        NUMERICAL_STRUCT_ARRAY, 'a', False, 3, True)
    assert np.array_equal(interpolated_values, values)
    assert steps_number == steps

    # Numerical interpolations with mixed array
    values = np.array([0., 0.5, 1.])
    interpolated_values, steps_number = ftmfi._get_feature_range(
        MIXED_ARRAY, 'a', False, 3, True)
    assert np.array_equal(interpolated_values, values)
    assert steps_number == steps

    # Categorical interpolation with classic array
    values = np.array(['b', 'c', 'f'])
    interpolated_values, steps_number = ftmfi._get_feature_range(
        CATEGORICAL_NP_ARRAY, 1, True, 3, False)
    assert np.array_equal(interpolated_values, values)
    assert steps_number == steps

    # Categorical interpolation with classic array and wrong step number
    interpolated_values, steps_number = ftmfi._get_feature_range(
        CATEGORICAL_NP_ARRAY, 1, True, 100, False)
    assert np.array_equal(interpolated_values, values)
    assert steps_number == steps

    # Categorical interpolation with structured array
    interpolated_values, steps_number = ftmfi._get_feature_range(
        CATEGORICAL_STRUCT_ARRAY, 'b', True, 3, True)
    assert np.array_equal(interpolated_values, values)
    assert steps_number == steps

    # Categorical interpolation with mixed array
    values = np.array(['a', 'c', 'f'])
    interpolated_values, steps_number = ftmfi._get_feature_range(
        MIXED_ARRAY, 'b', True, 3, True)
    assert np.array_equal(interpolated_values, values)
    assert steps_number == steps


def test_interpolate_array():
    """
    Tests array interpolation.

    This function tests
    :func:`fatf.transparency.models.feature_influence._interpolate_array`.
    """
    # For a structured and an unstructured *numerical* arrays...
    feature_index_num = 1
    feature_index_cat = 'b'
    #
    num_1_min = 0
    num_1_max = 1
    num_1_unique = np.array([num_1_min, num_1_max])
    cat_1_unique = np.array(['b', 'c', 'f'])
    #
    sar1 = NUMERICAL_NP_ARRAY.copy()
    sar1[:, feature_index_num] = num_1_min
    sar2 = NUMERICAL_NP_ARRAY.copy()
    sar2[:, feature_index_num] = num_1_max
    num_1_data_unique = np.stack([sar1, sar2], axis=1)
    #
    num_1_interpolate_3 = np.array([num_1_min, 0.5, num_1_max])
    #
    sar = []
    for i in num_1_interpolate_3:
        sar_i = NUMERICAL_NP_ARRAY.copy()
        sar_i[:, feature_index_num] = i
        sar.append(sar_i)
    num_1_data_interpolate_3 = np.stack(sar, axis=1)
    #
    sar = []
    for i in cat_1_unique:
        sar_i = CATEGORICAL_NP_ARRAY.copy()
        sar_i[:, feature_index_num] = i
        sar.append(sar_i)
    cat_1_interpolate = np.stack(sar, axis=1)
    # ...treat a numerical feature as a categorical one
    # ......with default steps number (without)
    interpolated_data, interpolated_values = ftmfi._interpolate_array(
        NUMERICAL_NP_ARRAY, feature_index_num, True, None)
    assert np.array_equal(interpolated_data, num_1_data_unique)
    assert np.array_equal(interpolated_values, num_1_unique)
    # ......with steps number
    interpolated_data, interpolated_values = ftmfi._interpolate_array(
        NUMERICAL_NP_ARRAY, feature_index_num, True, 3)
    assert np.array_equal(interpolated_data, num_1_data_unique)
    assert np.array_equal(interpolated_values, num_1_unique)

    # ...treat a numerical feature as a numerical one
    # ......with default steps number (without) -- this cannot be achieved
    pass
    # ......with steps number
    interpolated_data, interpolated_values = ftmfi._interpolate_array(
        NUMERICAL_STRUCT_ARRAY, feature_index_cat, False, 3)
    for index, column in enumerate(NUMERICAL_STRUCT_ARRAY.dtype.names):
        assert np.allclose(interpolated_data[:, :][column],
                           num_1_data_interpolate_3[:, :, index])
    assert np.array_equal(interpolated_values, num_1_interpolate_3)

    # ...treat a categorical feature as a categorical one
    # ......with default steps number (without)
    interpolated_data, interpolated_values = ftmfi._interpolate_array(
        CATEGORICAL_NP_ARRAY, feature_index_num, True, None)
    assert np.array_equal(interpolated_data, cat_1_interpolate)
    assert np.array_equal(interpolated_values, cat_1_unique)
    # ......with steps number
    interpolated_data, interpolated_values = ftmfi._interpolate_array(
        CATEGORICAL_STRUCT_ARRAY, feature_index_cat, True, 3)
    for index, column in enumerate(CATEGORICAL_STRUCT_ARRAY.dtype.names):
        assert np.array_equal(interpolated_data[:, :][column],
                              cat_1_interpolate[:, :, index])
    assert np.array_equal(interpolated_values, cat_1_unique)

    # ...treat a categorical feature as a numerical one
    # ......with default steps number (without)
    pass
    # ......with steps number
    pass

    ###########################################################################

    numerical_column = 'a'
    numreical_linespace_cat = np.array([0, 1])
    sar = []
    for i in numreical_linespace_cat:
        sar_i = MIXED_ARRAY.copy()
        sar_i[numerical_column] = i
        sar.append(sar_i)
    numerical_interpolation_cat = np.stack(sar, axis=1)
    #
    numreical_linespace_num = np.array([0, 0.5, 1])
    sar = []
    for i in numreical_linespace_num:
        # Redo the type
        dtype = [(name, numreical_linespace_num.dtype)
                 if name == numerical_column
                 else (name, MIXED_ARRAY.dtype[name])
                 for name in MIXED_ARRAY.dtype.names]  # yapf: disable
        sar_i = MIXED_ARRAY.astype(dtype)

        sar_i[numerical_column] = i
        sar.append(sar_i)
    numerical_interpolation_num = np.stack(sar, axis=1)

    categorical_column = 'b'
    categorical_linespace = np.array(['a', 'c', 'f'])
    sar = []
    for i in categorical_linespace:
        sar_i = MIXED_ARRAY.copy()
        sar_i[categorical_column] = i
        sar.append(sar_i)
    categorical_interpolation = np.stack(sar, axis=1)

    # Now for a mixed structured array -- categorical feature
    # ...treat a categorical feature as a categorical one
    # ......with default steps number (without)
    interpolated_data, interpolated_values = ftmfi._interpolate_array(
        MIXED_ARRAY, categorical_column, True, None)
    assert np.array_equal(interpolated_values, categorical_linespace)
    for column in MIXED_ARRAY.dtype.names:
        assert np.array_equal(interpolated_data[:, :][column],
                              categorical_interpolation[:, :][column])
    # ......with steps number
    interpolated_data, interpolated_values = ftmfi._interpolate_array(
        MIXED_ARRAY, categorical_column, True, 42)
    assert np.array_equal(interpolated_values, categorical_linespace)
    for column in MIXED_ARRAY.dtype.names:
        assert np.array_equal(interpolated_data[:, :][column],
                              categorical_interpolation[:, :][column])

    # Now for a mixed structured array -- numerical feature
    # ...treat a numerical feature as a categorical one
    # ......with default steps number (without)
    interpolated_data, interpolated_values = ftmfi._interpolate_array(
        MIXED_ARRAY, numerical_column, True, None)
    assert np.array_equal(interpolated_values, numreical_linespace_cat)
    for column in MIXED_ARRAY.dtype.names:
        assert np.array_equal(interpolated_data[:, :][column],
                              numerical_interpolation_cat[:, :][column])
    # ......with steps number
    interpolated_data, interpolated_values = ftmfi._interpolate_array(
        MIXED_ARRAY, numerical_column, True, 3)
    assert np.array_equal(interpolated_values, numreical_linespace_cat)
    for column in MIXED_ARRAY.dtype.names:
        assert np.array_equal(interpolated_data[:, :][column],
                              numerical_interpolation_cat[:, :][column])
    # ...treat a numerical feature as a numerical one
    # ......with steps number
    interpolated_data, interpolated_values = ftmfi._interpolate_array(
        MIXED_ARRAY, numerical_column, False, 3)
    assert np.array_equal(interpolated_values, numreical_linespace_num)
    for column in MIXED_ARRAY.dtype.names:
        assert np.array_equal(interpolated_data[:, :][column],
                              numerical_interpolation_num[:, :][column])


def test_interpolate_array_2d():
    """
    Tests array interpolation across two features.

    This function tests
    :func:`fatf.transparency.models.feature_influence._interpolate_array_2d`.
    """
    # FOR CLASSIC ARRAYS
    # For a structured and an unstructured *numerical* arrays...
    numerical_columns = [0, 1]
    struct_columns = ['a', 'b']
    numerical_linespace = np.array([[0., 1., 2.,], [0., 0.5, 1.]])
    numerical_linespace_100 = [np.array([0., 1., 2.]), np.linspace(0., 1., 100)]
    numerical_cat_linespace = [np.array([0., 1., 2.]), np.array([0., 1.])]
    numerical_interpolate_1 = np.array(6*[[[0.]*3, [1.]*3, [2.]*3]])
    numerical_interpolate_cat_1 = np.array(6*[[[0.]*2, [1.]*2, [2.]*2]])
    numerical_interpolate_1_100 = np.array(6*[[[0.]*100, [1.]*100, [2.]*100]])
    numerical_interpolate_2 = np.array(6*[[[0., 0.5, 1]]*3])
    numerical_interpolate_cat_2 = np.array(6*[[[0., 1.]]*3])
    numerical_interpolate_2_100 = np.array(6*[[np.linspace(0., 1., 100)]*3])

    categorical_linespace = [np.array(['a', 'b']), np.array(['b', 'c', 'f'])]
    categorical_interpolate_1 = np.array(3*[[['a']*3, ['b']*3]])
    categorical_interpolate_2 = np.array(3*[[['b', 'c', 'f']]*2])

    mixed_columns_num = ['a', 'c']
    mixed_columns_mix = ['a', 'b']
    mixed_columns_str = ['b', 'd']
    mixed_linespace_num = [np.array([0., 0.5, 1.]),
                           np.linspace(0.07, 0.99, 100)]
    mixed_linespace_mix = [np.array([0., 0.5, 1.]), np.array(['a', 'c', 'f'])]
    mixed_linespace_str = [np.array(['a', 'c', 'f']),
                           np.array(['a', 'aa', 'b', 'bb'])]
    mixed_interpolate_1_num = np.array(6*[[[0.]*100, [0.5]*100, [1.]*100]])
    mixed_interpolate_2_num = np.array(6*[[np.linspace(0.07, 0.99, 100)]*3])
    mixed_interpolate_1_mix = np.array(6*[[[0.]*3, [0.5]*3, [1.]*3]])
    mixed_interpolate_2_mix = np.array(6*[[['a', 'c', 'f']]*3])
    mixed_interpolate_1_str = np.array(6*[[['a']*4, ['c']*4, ['f']*4]])
    mixed_interpolate_2_str = np.array(6*[[['a', 'aa', 'b', 'bb']]*3])

    # Treat both columns as numerical with step size 3
    interpolated_data, interpolated_values = ftmfi._interpolate_array_2d(
        NUMERICAL_NP_ARRAY, numerical_columns, [False, False], [3, 3])
    assert np.array_equal(numerical_linespace, 
                          np.stack(interpolated_values, axis=0))
    assert np.array_equal(numerical_interpolate_1,
                          interpolated_data[:, :, :, numerical_columns[0]])
    assert np.array_equal(numerical_interpolate_2,
                          interpolated_data[:, :, :, numerical_columns[1]])

    # Treat second column as catageorical
    interpolated_data, interpolated_values = ftmfi._interpolate_array_2d(
        NUMERICAL_NP_ARRAY, numerical_columns, [False, True], [3, None])
    for true_value, interp_value in zip(numerical_cat_linespace, 
            interpolated_values):
        assert np.array_equal(true_value, interp_value)
    assert np.array_equal(numerical_interpolate_cat_1,
                          interpolated_data[:, :, :, numerical_columns[0]])
    assert np.array_equal(numerical_interpolate_cat_2,
                          interpolated_data[:, :, :, numerical_columns[1]])

    # Treat both as categorical
    interpolated_data, interpolated_values = ftmfi._interpolate_array_2d(
        NUMERICAL_NP_ARRAY, numerical_columns, [True, True], [3, None])
    for true_value, interp_value in zip(numerical_cat_linespace, 
            interpolated_values):
        assert np.array_equal(true_value, interp_value)
    assert np.array_equal(numerical_interpolate_cat_1,
                          interpolated_data[:, :, :, numerical_columns[0]])
    assert np.array_equal(numerical_interpolate_cat_2,
                          interpolated_data[:, :, :, numerical_columns[1]])

    # Steps sizes [3, 100] for purely numerical non-categorical
    interpolated_data, interpolated_values = ftmfi._interpolate_array_2d(
        NUMERICAL_NP_ARRAY, numerical_columns, [False, False], [3, 100])
    for true_value, interp_value in zip(numerical_linespace_100, 
            interpolated_values):
        assert np.array_equal(true_value, interp_value)
    assert np.array_equal(numerical_interpolate_1_100,
                          interpolated_data[:, :, :, numerical_columns[0]])
    assert np.array_equal(numerical_interpolate_2_100,
                          interpolated_data[:, :, :, numerical_columns[1]])

    # String data with strings
    interpolated_data, interpolated_values = ftmfi._interpolate_array_2d(
        CATEGORICAL_NP_ARRAY, numerical_columns, [True, True], [2, 3])
    for true_value, interp_value in zip(categorical_linespace, 
            interpolated_values):
        assert np.array_equal(true_value, interp_value)
    assert np.array_equal(categorical_interpolate_1,
                          interpolated_data[:, :, :, numerical_columns[0]])
    assert np.array_equal(categorical_interpolate_2,
                          interpolated_data[:, :, :, numerical_columns[1]])

    # String data with wrong number of steps
    interpolated_data, interpolated_values = ftmfi._interpolate_array_2d(
        CATEGORICAL_NP_ARRAY, numerical_columns, [True, True], [100, 100])
    for true_value, interp_value in zip(categorical_linespace, 
            interpolated_values):
        assert np.array_equal(true_value, interp_value)
    assert np.array_equal(categorical_interpolate_1,
                          interpolated_data[:, :, :, numerical_columns[0]])
    assert np.array_equal(categorical_interpolate_2,
                          interpolated_data[:, :, :, numerical_columns[1]])

    ###########################################################################

    # STRUCTURED ARRAYS
    # Structured with just numerical values
    interpolated_data, interpolated_values = ftmfi._interpolate_array_2d(
        NUMERICAL_STRUCT_ARRAY, struct_columns, [False, False], [3, 3])
    # When we go from int -> float in _generalise_dataset_dtype it
    # introduces a small error in the values
    assert np.allclose(numerical_linespace, interpolated_values, atol=1e-1)
    assert np.allclose(numerical_interpolate_1,
                          interpolated_data[:, :, :][struct_columns[0]])
    assert np.allclose(numerical_interpolate_2,
                       interpolated_data[:, :, :][struct_columns[1]],
                       atol=1e-1)
    
    # Treat second column as categorical
    interpolated_data, interpolated_values = ftmfi._interpolate_array_2d(
        NUMERICAL_STRUCT_ARRAY, struct_columns, [False, True], [3, None])
    for true_value, interp_value in zip(numerical_cat_linespace, 
            interpolated_values):
        assert np.allclose(true_value, interp_value, atol=1e-1)
    assert np.allclose(numerical_interpolate_cat_1,
                          interpolated_data[:, :, :][struct_columns[0]])
    assert np.allclose(numerical_interpolate_cat_2,
                       interpolated_data[:, :, :][struct_columns[1]],
                       atol=1e-1)

    # Steps sizes [3, 100] for purely numerical non-categorical
    interpolated_data, interpolated_values = ftmfi._interpolate_array_2d(
        NUMERICAL_STRUCT_ARRAY, struct_columns, [False, False], [3, 100])
    for true_value, interp_value in zip(numerical_linespace_100, 
            interpolated_values):
        assert np.allclose(true_value, interp_value, atol=1e-1)
    assert np.allclose(numerical_interpolate_1_100,
                          interpolated_data[:, :, :][struct_columns[0]])
    assert np.allclose(numerical_interpolate_2_100,
                          interpolated_data[:, :, :][struct_columns[1]],
                          atol=1e-1)
    
    # String data with strings
    interpolated_data, interpolated_values = ftmfi._interpolate_array_2d(
        CATEGORICAL_STRUCT_ARRAY, struct_columns, [True, True], [2, 3])
    for true_value, interp_value in zip(categorical_linespace, 
            interpolated_values):
        assert np.array_equal(true_value, interp_value)
    assert np.array_equal(categorical_interpolate_1,
                          interpolated_data[:, :, :][struct_columns[0]])
    assert np.array_equal(categorical_interpolate_2,
                          interpolated_data[:, :, :][struct_columns[1]])

    # String data with wrong number of steps
    interpolated_data, interpolated_values = ftmfi._interpolate_array_2d(
        CATEGORICAL_STRUCT_ARRAY, struct_columns, [True, True], [100, 100])
    for true_value, interp_value in zip(categorical_linespace, 
            interpolated_values):
        assert np.array_equal(true_value, interp_value)
    assert np.array_equal(categorical_interpolate_1,
                          interpolated_data[:, :, :][struct_columns[0]])
    assert np.array_equal(categorical_interpolate_2,
                          interpolated_data[:, :, :][struct_columns[1]])

    ###########################################################################

    # MIXED ARRAYS
    # Interpolate both numerical values
    interpolated_data, interpolated_values = ftmfi._interpolate_array_2d(
        MIXED_ARRAY, mixed_columns_num, [False, False], [3, 100])
    for true_value, interp_value in zip(mixed_linespace_num, 
            interpolated_values):
        assert np.allclose(true_value, interp_value, atol=1e-1)
    assert np.allclose(mixed_interpolate_1_num,
                       interpolated_data[:, :, :][mixed_columns_num[0]])
    assert np.allclose(mixed_interpolate_2_num,
                       interpolated_data[:, :, :][mixed_columns_num[1]],
                       atol=1e-1)

    # Interpolate one numerical one string
    interpolated_data, interpolated_values = ftmfi._interpolate_array_2d(
        MIXED_ARRAY, mixed_columns_mix, [False, True], [3, None])
    assert np.allclose(interpolated_values[0], mixed_linespace_mix[0],
                       atol=1e-1)
    assert np.array_equal(interpolated_values[1], mixed_linespace_mix[1])
    assert np.allclose(mixed_interpolate_1_mix,
                       interpolated_data[:, :, :][mixed_columns_mix[0]],
                       atol=1e-1)
    assert np.array_equal(mixed_interpolate_2_mix,
                          interpolated_data[:, :, :][mixed_columns_mix[1]])

    # Interpolate both string
    interpolated_data, interpolated_values = ftmfi._interpolate_array_2d(
        MIXED_ARRAY, mixed_columns_str, [True, True], [None, None])
    for true_value, interp_value in zip(mixed_linespace_str, 
            interpolated_values):
        assert np.array_equal(true_value, interp_value)
    assert np.array_equal(mixed_interpolate_1_str,
                          interpolated_data[:, :, :][mixed_columns_str[0]])
    assert np.array_equal(mixed_interpolate_2_str,
                          interpolated_data[:, :, :][mixed_columns_str[1]])


def test_infer_is_categorical_steps_number():
    user_warning = ('Selected feature is categorical (string-base elements), '
                    'however the treat_as_categorical was set to False. Such '
                    'a combination is not possible. The feature will be '
                    'treated as categorical.')
    steps_n_warning = ('The steps_number parameter will be ignored as the '
                       'feature is being treated as categorical.')

    with pytest.warns(UserWarning) as warning:
        treat_as_categorical, steps_number = \
            ftmfi._infer_is_categorical_steps_number(
                NUMERICAL_NP_ARRAY_TEST[:, 0], True, 10)
    assert len(warning) == 1
    assert str(warning[0].message) == steps_n_warning
    assert treat_as_categorical
    assert steps_number is 10

    with pytest.warns(UserWarning) as warning:
        treat_as_categorical, steps_number = \
            ftmfi._infer_is_categorical_steps_number(
                CATEGORICAL_NP_ARRAY_TEST[:, 0], False, None)
    assert len(warning) == 1
    assert str(warning[0].message) == user_warning
    assert treat_as_categorical
    assert steps_number is None

    # Working fine
    treat_as_categorical, steps_number = \
        ftmfi._infer_is_categorical_steps_number(
            NUMERICAL_NP_ARRAY_TEST[:, 0], False, 10)
    assert not treat_as_categorical
    assert steps_number == 10

    treat_as_categorical, steps_number = \
        ftmfi._infer_is_categorical_steps_number(
            NUMERICAL_NP_ARRAY_TEST[:, 0], False, None)
    assert not treat_as_categorical
    assert steps_number == 100

    treat_as_categorical, steps_number = \
        ftmfi._infer_is_categorical_steps_number(
            NUMERICAL_NP_ARRAY_TEST[:, 0], True, None)
    assert treat_as_categorical
    assert steps_number is None

    treat_as_categorical, steps_number = \
        ftmfi._infer_is_categorical_steps_number(
            CATEGORICAL_NP_ARRAY_TEST[:, 0], True, None)
    assert treat_as_categorical
    assert steps_number is None


def test_filter_rows():
    """
    Tests :func:`fatf.transparency.models.feature_influence._filter_rows`.
    """
    value_error = ('{} rows element {} is out of bounds. There are only {} '
                   'rows in the input dataset.')
    type_error_include = ('The include_rows parameters must be either None or '
                          'a list of integers indicating which rows should be '
                          'included in the computation.')
    type_error_include_list = 'Include rows element *{}* is not an integer.'
    type_error_exclude = ('The exclude_rows parameters must be either None or '
                          'a list of integers indicating which rows should be '
                          'excluded in the computation.')
    type_error_exclude_list = 'Exclude rows element *{}* is not an integer.'

    with pytest.raises(TypeError) as exin:
        ftmfi._filter_rows('wrong', None, 1)
    assert str(exin.value) == type_error_include
    with pytest.raises(TypeError) as exin:
        ftmfi._filter_rows([0, 1, 'wrong', 4, 5], None, 7)
    assert str(exin.value) == type_error_include_list.format('wrong')
    with pytest.raises(TypeError) as exin:
        ftmfi._filter_rows(None, 'wrong', 1)
    assert str(exin.value) == type_error_exclude
    with pytest.raises(TypeError) as exin:
        ftmfi._filter_rows(None, [0, 1, 'wrong', 4, 5], 7)
    assert str(exin.value) == type_error_exclude_list.format('wrong')

    with pytest.raises(ValueError) as exin:
        ftmfi._filter_rows(None, [0, 1, 3, 5], 4)
    assert str(exin.value) == value_error.format('Exclude', 5, 4)
    with pytest.raises(ValueError) as exin:
        ftmfi._filter_rows(None, 5, 4)
    assert str(exin.value) == value_error.format('Exclude', 5, 4)
    with pytest.raises(ValueError) as exin:
        ftmfi._filter_rows([0, 1, 3, 5], None, 4)
    assert str(exin.value) == value_error.format('Include', 5, 4)
    with pytest.raises(ValueError) as exin:
        ftmfi._filter_rows(5, None, 4)
    assert str(exin.value) == value_error.format('Include', 5, 4)

    row_number = 13
    row_none = None
    row_digit = 3
    row_list = [3, 4, 7, 12]

    all_rows = list(range(13))
    all_but_one = [0, 1, 2] + list(range(4, 13))
    all_but_list = [0, 1, 2, 5, 6, 8, 9, 10, 11]
    row_but_one = [4, 7, 12]
    three = [3]
    empty = []

    rows = ftmfi._filter_rows(row_none, row_none, row_number)
    assert np.array_equal(rows, all_rows)
    rows = ftmfi._filter_rows(row_none, row_digit, row_number)
    assert np.array_equal(rows, all_but_one)
    rows = ftmfi._filter_rows(row_none, row_list, row_number)
    assert np.array_equal(rows, all_but_list)
    rows = ftmfi._filter_rows(row_none, empty, row_number)
    assert np.array_equal(rows, all_rows)
    rows = ftmfi._filter_rows(empty, row_none, row_number)
    assert np.array_equal(rows, empty)

    rows = ftmfi._filter_rows(row_digit, row_none, row_number)
    assert np.array_equal(rows, three)
    rows = ftmfi._filter_rows(row_digit, row_digit, row_number)
    assert np.array_equal(rows, empty)
    rows = ftmfi._filter_rows(row_digit, row_list, row_number)
    assert np.array_equal(rows, empty)
    rows = ftmfi._filter_rows(row_digit, empty, row_number)
    assert np.array_equal(rows, three)
    rows = ftmfi._filter_rows(empty, row_digit, row_number)
    assert np.array_equal(rows, empty)

    rows = ftmfi._filter_rows(row_list, row_none, row_number)
    assert np.array_equal(rows, row_list)
    rows = ftmfi._filter_rows(row_list, row_digit, row_number)
    assert np.array_equal(rows, row_but_one)
    rows = ftmfi._filter_rows(row_list, row_list, row_number)
    assert np.array_equal(rows, empty)
    rows = ftmfi._filter_rows(row_list, empty, row_number)
    assert np.array_equal(rows, row_list)
    rows = ftmfi._filter_rows(empty, row_list, row_number)
    assert np.array_equal(rows, empty)


def test_merge_ice_arrays():
    """
    Tests :func:`fatf.transparency.models.feature_influence.merge_ice_arrays`.
    """
    type_error = ('The ice_arrays_list should be a list of numpy arrays that '
                  'represent Individual Conditional Expectation.')
    value_error_empty = 'Cannot merge 0 arrays.'
    value_error_numerical = ('The ice_array list should only contain '
                             'numerical arrays.')
    value_error_struct = ('The ice_array list should only contain '
                          'unstructured arrays.')
    incorrect_shape_3d = ('The ice_array should be 3-dimensional or '
                          '4-dimensional for 2 feature ICE.')
    value_error_shape = ('All of the ICE arrays need to be constructed for '
                         'the same number of classes and the same number of '
                         'samples for the selected feature (the second and '
                         'the third dimension of the ice array).')

    with pytest.raises(TypeError) as exin:
        ftmfi.merge_ice_arrays('string')
    assert str(exin.value) == type_error
    with pytest.raises(ValueError) as exin:
        ftmfi.merge_ice_arrays([])
    assert str(exin.value) == value_error_empty
    with pytest.raises(ValueError) as exin:
        ftmfi.merge_ice_arrays([np.array([1, 2, 'a', 4, 5])])
    assert str(exin.value) == value_error_numerical
    with pytest.raises(ValueError) as exin:
        ftmfi.merge_ice_arrays(
            [np.array([[[4]]]),
             np.array([(1, )], dtype=[('a', int)])])
    assert str(exin.value) == value_error_struct
    with pytest.raises(IncorrectShapeError) as exin:
        ftmfi.merge_ice_arrays([np.array([[[4]]]), np.array([2])])
    assert str(exin.value) == incorrect_shape_3d

    arr_1 = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 9, 8]],
                      [[7, 6, 5, 4], [3, 2, 1, 0], [1, 2, 3, 4]]])
    arr_2 = np.array([[[7, 6, 5], [3, 2, 1], [1, 2, 3]]])
    arr_3 = np.array([[[7, 6, 5, 4], [3, 2, 1, 0]]])
    arr_4 = np.array([[[7, 6, 5, 4], [3, 2, 1, 0]]], dtype=float)
    with pytest.raises(ValueError) as exin:
        ftmfi.merge_ice_arrays([arr_1, arr_1, arr_2])
    assert str(exin.value) == value_error_shape
    with pytest.raises(ValueError) as exin:
        ftmfi.merge_ice_arrays([arr_1, arr_3, arr_2])
    assert str(exin.value) == value_error_shape
    with pytest.raises(ValueError) as exin:
        ftmfi.merge_ice_arrays([arr_3, arr_3, arr_4])
    assert str(exin.value) == value_error_shape

    # Unstructured ICE arrays
    selected_column_index = 1
    smaller_numerical_array = np.array([[0, 0, 0.08, 0.69],
                                        [1, 0, 0.03, 0.29],
                                        [0, 1, 0.99, 0.82]])  # yapf: disable
    concat = np.concatenate([NUMERICAL_NP_ARRAY, smaller_numerical_array])
    arr_a = []
    arr_b = []
    arr_c = []
    for i in range(3):
        arr_i = NUMERICAL_NP_ARRAY.copy()
        arr_i[:, selected_column_index] = i
        arr_a.append(arr_i)

        arr_i = smaller_numerical_array.copy()
        arr_i[:, selected_column_index] = i
        arr_b.append(arr_i)

        arr_i = concat.copy()
        arr_i[:, selected_column_index] = i
        arr_c.append(arr_i)
    unstructured_array_a = np.stack(arr_a, axis=1)
    unstructured_array_b = np.stack(arr_b, axis=1)
    unstructured_array_c = np.stack(arr_c, axis=1)

    comp = ftmfi.merge_ice_arrays([unstructured_array_a, unstructured_array_b])
    assert np.array_equal(comp, unstructured_array_c)

    # 2-D ICE arrays
    unstructured_array_a = np.repeat(unstructured_array_a[:, :, np.newaxis, :],
                                     5, axis=2)
    unstructured_array_b = np.repeat(unstructured_array_b[:, :, np.newaxis, :],
                                     5, axis=2)
    unstructured_array_c = np.repeat(unstructured_array_c[:, :, np.newaxis, :],
                                     5, axis=2)
    comp = ftmfi.merge_ice_arrays([unstructured_array_a, unstructured_array_b])
    assert np.array_equal(comp, unstructured_array_c)


def test_compute_feature_distribution():
    """
    Tests :func:`fatf.transparency.models.feature_influence.
    compute_feature_distribution` function.
    """
    fatf.setup_random_seed()
    msg = ('kde must be a boolean.')
    with pytest.raises(AssertionError) as exin:
        ftmfi.compute_feature_distribution(None, None, False, 1, None)
    assert str(exin.value) == msg

    msg = ('samples must be an integer.')
    with pytest.raises(AssertionError) as exin:
        ftmfi.compute_feature_distribution(None, None, False, False, '1')
    assert str(exin.value) == msg

    msg = ('treat_as_categorical was set to True and kde was set to True. '
           'Gaussian kernel estimation cannot be used on categorical data.')
    with pytest.raises(ValueError) as exin:
        ftmfi.compute_feature_distribution(NUMERICAL_NP_ARRAY, 0, True, True)
    assert str(exin.value) == msg

    msg = ('Selected feature is categorical (string-base elements), however '
           'kde was set to True. Gaussian kernel estimation cannot be used on '
           'categorical data.')
    with pytest.raises(ValueError) as exin:
        ftmfi.compute_feature_distribution(CATEGORICAL_STRUCT_ARRAY, 'c',
                                       treat_as_categorical=False, kde=True)
    assert str(exin.value) == msg

    msg = ('Selected feature is categorical (string-base elements), however '
           'the treat_as_categorical was set to False. Such a combination is '
           'not possible. The feature will be treated as categorical.')
    with pytest.warns(UserWarning) as warning:
        ftmfi.compute_feature_distribution(CATEGORICAL_STRUCT_ARRAY, 'c', 
                                      treat_as_categorical=False)
    assert len(warning) == 1
    assert str(warning[0].message) == msg

    msg = ('The samples parameter will be ignored as the feature is '
            'being treated as categorical. The number of bins will be the '
            'number of unique values in the feature.')
    with pytest.warns(UserWarning) as warning:
        ftmfi.compute_feature_distribution(CATEGORICAL_STRUCT_ARRAY, 'c',
                                       treat_as_categorical=True, samples=10)
    assert len(warning) == 1
    assert str(warning[0].message) == msg

    # Categorical numerical (counts)
    values = np.array([0., 1., 2.,])
    counts = np.array([3., 2., 1.])
    counts = counts / np.sum(counts)
    dist = ftmfi.compute_feature_distribution(NUMERICAL_NP_ARRAY, 0,
                                          treat_as_categorical=True)
    assert np.array_equal(dist[0], values)
    assert np.array_equal(dist[1], counts)

    # Structured categorical numerical (counts)
    dist = ftmfi.compute_feature_distribution(NUMERICAL_STRUCT_ARRAY, 'a',
                                          treat_as_categorical=True)
    assert np.array_equal(dist[0], values)
    assert np.array_equal(dist[1], counts)

    # Categorical string (counts)
    values = np.array(['a', 'b'])
    counts = np.array([2., 1.])
    counts = counts / np.sum(counts)
    dist = ftmfi.compute_feature_distribution(CATEGORICAL_NP_ARRAY, 0,
                                          treat_as_categorical=True)
    assert np.array_equal(dist[0], values)
    assert np.array_equal(dist[1], counts)

    # Structured categorical string (counts)
    dist = ftmfi.compute_feature_distribution(CATEGORICAL_STRUCT_ARRAY, 'a',
                                          treat_as_categorical=True)
    assert np.array_equal(dist[0], values)
    assert np.array_equal(dist[1], counts)

    # Non-categorical numerical (histogram)
    values = np.array([0., 0.4, 0.8, 1.2, 1.6, 2.])
    counts = np.array([1.25, 0., 0.83, 0., 0.41])
    counts = counts / np.sum(counts)
    dist = ftmfi.compute_feature_distribution(NUMERICAL_NP_ARRAY, 0, samples=5)
    assert np.allclose(dist[0], values, atol=1e-2)
    assert np.allclose(dist[1], counts, atol=1e-2)

    # Structured non-categorical numerical (histogram)
    dist = ftmfi.compute_feature_distribution(NUMERICAL_STRUCT_ARRAY, 'a',
                                         samples=5)
    assert np.allclose(dist[0], values, atol=1e-2)
    assert np.allclose(dist[1], counts, atol=1e-2)

    # Non-categorical numerical (kde)
    values = np.array([0., 0.5, 1., 1.5, 2.])
    counts = np.array([0.400, 0.400, 0.333, 0.249, 0.167])
    dist = ftmfi.compute_feature_distribution(NUMERICAL_NP_ARRAY, 0, kde=True,
                                          samples=5)
    assert np.array_equal(dist[0], values)
    assert np.allclose(dist[1], counts, atol=1e-2)

    # Structured non-categorical numerical (kde)
    dist = ftmfi.compute_feature_distribution(NUMERICAL_STRUCT_ARRAY, 'a',
                                          kde=True, samples=5)
    assert np.array_equal(dist[0], values)
    assert np.allclose(dist[1], counts, atol=1e-2)

    # Non-categorical numerical (auto bins)
    dist = ftmfi.compute_feature_distribution(NUMERICAL_NP_ARRAY, 0)
    assert dist[0].min() == NUMERICAL_NP_ARRAY[:, 0].min()
    assert dist[0].max() == NUMERICAL_NP_ARRAY[:, 0].max()
    assert dist[0].shape[0] == dist[1].shape[0] + 1

    # Structured non-categorical numerical (auto bins)
    dist = ftmfi.compute_feature_distribution(NUMERICAL_STRUCT_ARRAY, 'a')
    assert dist[0].min() == NUMERICAL_NP_ARRAY[:, 0].min()
    assert dist[0].max() == NUMERICAL_NP_ARRAY[:, 0].max()
    assert dist[0].shape[0] == dist[1].shape[0] + 1


def test_individual_conditional_expectation():
    """
    Tests Individual Conditional Expectation calculations.

    Tests :func:`fatf.transparency.models.feature_influence.
    individual_conditional_expectation` function.
    """
    user_warning = ('Selected feature is categorical (string-base elements), '
                    'however the treat_as_categorical was set to False. Such '
                    'a combination is not possible. The feature will be '
                    'treated as categorical.')
    steps_n_warning = ('The steps_number parameter will be ignored as the '
                       'feature is being treated as categorical.')

    # Model
    msg = ('This functionality requires the classification model to be '
           'capable of outputting probabilities via predict_proba method.')
    model = InvalidModel()
    with pytest.warns(UserWarning) as warning:
        with pytest.raises(IncompatibleModelError) as exin:
            ftmfi.individual_conditional_expectation(
                NUMERICAL_NP_ARRAY, model, 0, 'classifier')
        assert str(exin.value) == msg
    assert len(warning) == 1
    assert str(warning[0].message) == ('The model class is missing '
                                       "'predict_proba' method.")

    msg = ('This functionaility requires the regression model to be '
           'capable of outputting predictions via predict method')
    model = InvalidModelRegression()
    with pytest.warns(UserWarning) as warning:
        with pytest.raises(IncompatibleModelError) as exin:
            ftmfi.individual_conditional_expectation(
                NUMERICAL_NP_ARRAY, model, 0, 'regressor', None)
        assert str(exin.value) == msg
    assert len(warning) == 1
    assert str(warning[0].message) == ('The model class is missing '
                                       "'predict' method.")

    msg = ('Mode {} is not a valid mode. Mode should be \'classifier\' for '
           'classification model or \'regressor\' for regression model.')
    with pytest.raises(ValueError) as exin:
        ftmfi.individual_conditional_expectation(
            NUMERICAL_NP_ARRAY, model, 0, 'mode')
    assert str(exin.value) == msg.format('mode')

    clf = fum.KNN(k=2)
    clf.fit(NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY_TARGET)
    clf_struct = fum.KNN(k=2)
    clf_struct.fit(NUMERICAL_STRUCT_ARRAY, NUMERICAL_NP_ARRAY_TARGET)

    # Test for type generalisation int -> float for classic arrays
    ice, linespace = ftmfi.individual_conditional_expectation(
        NUMERICAL_NP_ARRAY_TEST_INT,
        clf,
        0,
        treat_as_categorical=False,
        steps_number=3)
    assert np.allclose(
        ice,
        np.array([[[0, 0, 1], [0.5, 0, 0.5], [1, 0, 0]],
                  [[0, 0, 1], [0.5, 0, 0.5], [1, 0, 0]]]))
    assert np.allclose(linespace, np.array([0, 0.5, 1]))

    # Not structured and structured -- numerical
    # ...numerical column
    # ......indicate as numerical
    # .........with a steps number
    ice, linespace = ftmfi.individual_conditional_expectation(
        NUMERICAL_STRUCT_ARRAY_TEST,
        clf_struct,
        'd',
        treat_as_categorical=False,
        steps_number=3)
    assert np.allclose(ice, NUMERICAL_NP_ICE)
    assert np.allclose(linespace, NUMERICAL_NP_LINESPACE)
    # .........without a steps number
    ice, linespace = ftmfi.individual_conditional_expectation(
        NUMERICAL_NP_ARRAY_TEST, clf, 3, treat_as_categorical=False)
    assert np.allclose(ice, NUMERICAL_NP_ICE_100)
    assert np.allclose(linespace, NUMERICAL_NP_LINESPACE_100)
    # ......indicate as categorical
    # .........with a steps number
    with pytest.warns(UserWarning) as warning:
        ice, linespace = ftmfi.individual_conditional_expectation(
            NUMERICAL_NP_ARRAY_TEST,
            clf,
            3,
            treat_as_categorical=True,
            steps_number=3)
    assert len(warning) == 1
    assert str(warning[0].message) == steps_n_warning
    assert np.allclose(ice, NUMERICAL_NP_ICE_CAT)
    assert np.allclose(linespace, NUMERICAL_NP_LINESPACE_CAT)
    # .........without a steps number
    ice, linespace = ftmfi.individual_conditional_expectation(
        NUMERICAL_STRUCT_ARRAY_TEST,
        clf_struct,
        'd',
        treat_as_categorical=True)
    assert np.allclose(ice, NUMERICAL_NP_ICE_CAT)
    assert np.allclose(linespace, NUMERICAL_NP_LINESPACE_CAT)
    # ......indicate as None
    # .........with a steps number
    ice, linespace = ftmfi.individual_conditional_expectation(
        NUMERICAL_NP_ARRAY_TEST, clf, 3, steps_number=3)
    assert np.allclose(ice, NUMERICAL_NP_ICE)
    assert np.allclose(linespace, NUMERICAL_NP_LINESPACE)
    # .........without a steps number
    ice, linespace = ftmfi.individual_conditional_expectation(
        NUMERICAL_NP_ARRAY_TEST, clf, 3)
    assert np.allclose(ice, NUMERICAL_NP_ICE_100)
    assert np.allclose(linespace, NUMERICAL_NP_LINESPACE_100)

    clf = fum.KNN(k=2)
    clf.fit(CATEGORICAL_NP_ARRAY, CATEGORICAL_NP_ARRAY_TARGET)
    clf_struct = fum.KNN(k=2)
    clf_struct.fit(CATEGORICAL_STRUCT_ARRAY, CATEGORICAL_NP_ARRAY_TARGET)

    # Not structured and structured -- categorical
    # ...categorical column
    # ......indicate as numerical
    # .........with a steps number
    with pytest.warns(UserWarning) as warning:
        ice, linespace = ftmfi.individual_conditional_expectation(
            CATEGORICAL_STRUCT_ARRAY_TEST,
            clf_struct,
            'c',
            treat_as_categorical=False,
            steps_number=3)
    assert len(warning) == 1
    assert str(warning[0].message) == user_warning
    assert np.allclose(ice, CATEGORICAL_NP_ICE)
    assert np.array_equal(linespace, CATEGORICAL_NP_LINESPACE)
    # .........without a steps number
    with pytest.warns(UserWarning) as warning:
        ice, linespace = ftmfi.individual_conditional_expectation(
            CATEGORICAL_NP_ARRAY_TEST, clf, 2, treat_as_categorical=False)
    assert len(warning) == 1
    assert str(warning[0].message) == user_warning
    assert np.allclose(ice, CATEGORICAL_NP_ICE)
    assert np.array_equal(linespace, CATEGORICAL_NP_LINESPACE)
    # ......indicate as categorical
    # .........with a steps number
    with pytest.warns(UserWarning) as warning:
        ice, linespace = ftmfi.individual_conditional_expectation(
            CATEGORICAL_STRUCT_ARRAY_TEST,
            clf_struct,
            'c',
            treat_as_categorical=True,
            steps_number=42)
    assert len(warning) == 1
    assert str(warning[0].message) == steps_n_warning
    assert np.allclose(ice, CATEGORICAL_NP_ICE)
    assert np.array_equal(linespace, CATEGORICAL_NP_LINESPACE)
    # .........without a steps number
    ice, linespace = ftmfi.individual_conditional_expectation(
        CATEGORICAL_NP_ARRAY_TEST, clf, 2, treat_as_categorical=True)
    assert np.allclose(ice, CATEGORICAL_NP_ICE)
    assert np.array_equal(linespace, CATEGORICAL_NP_LINESPACE)
    # ......indicate as None
    # .........with a steps number
    with pytest.warns(UserWarning) as warning:
        ice, linespace = ftmfi.individual_conditional_expectation(
            CATEGORICAL_NP_ARRAY_TEST, clf, 2, steps_number=42)
    assert len(warning) == 1
    assert str(warning[0].message) == steps_n_warning
    assert np.allclose(ice, CATEGORICAL_NP_ICE)
    assert np.array_equal(linespace, CATEGORICAL_NP_LINESPACE)
    # .........without a steps number
    ice, linespace = ftmfi.individual_conditional_expectation(
        CATEGORICAL_STRUCT_ARRAY_TEST, clf_struct, 'c')
    assert np.allclose(ice, CATEGORICAL_NP_ICE)
    assert np.array_equal(linespace, CATEGORICAL_NP_LINESPACE)

    # Mixed array; include/exclude some rows
    clf = fum.KNN(k=2)
    clf.fit(MIXED_ARRAY, MIXED_ARRAY_TARGET)

    ice, linespace = ftmfi.individual_conditional_expectation(
        MIXED_ARRAY_TEST, clf, 'a', steps_number=3, exclude_rows=1)
    assert np.allclose(ice, MIXED_ICE_NUMERICAL)
    assert np.array_equal(linespace, MIXED_LINESPACE_NUMERICAL)

    ice, linespace = ftmfi.individual_conditional_expectation(
        MIXED_ARRAY_TEST,
        clf,
        'a',
        steps_number=3,
        include_rows=[0, 2],
        exclude_rows=[1])
    assert np.allclose(ice, MIXED_ICE_NUMERICAL)
    assert np.array_equal(linespace, MIXED_LINESPACE_NUMERICAL)

    ice, linespace = ftmfi.individual_conditional_expectation(
        MIXED_ARRAY_TEST,
        clf,
        'a',
        steps_number=3,
        include_rows=[0, 2],
        exclude_rows=1)
    assert np.allclose(ice, MIXED_ICE_NUMERICAL)
    assert np.array_equal(linespace, MIXED_LINESPACE_NUMERICAL)

    ice, linespace = ftmfi.individual_conditional_expectation(
        MIXED_ARRAY_TEST, clf, 'b', exclude_rows=1)
    assert np.allclose(ice, MIXED_ICE_CATEGORICAL)
    assert np.array_equal(linespace, MIXED_LINESPACE_CATEGORICAL)

    ice, linespace = ftmfi.individual_conditional_expectation(
        MIXED_ARRAY_TEST, clf, 'b', include_rows=[0, 2], exclude_rows=1)
    assert np.allclose(ice, MIXED_ICE_CATEGORICAL)
    assert np.array_equal(linespace, MIXED_LINESPACE_CATEGORICAL)

    # Test Regression
    clf = fum.KNN(k=2, mode='regressor')
    clf.fit(NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY_TARGET.astype(np.float32))

    ice, linespace = ftmfi.individual_conditional_expectation(
        NUMERICAL_NP_ARRAY_TEST, clf, 3, 'regressor', steps_number=3)
    assert np.allclose(ice, NUMERICAL_NP_ICE_REGRESSION)
    assert np.allclose(linespace, NUMERICAL_NP_LINESPACE)

    ice, linespace = ftmfi.individual_conditional_expectation(
        NUMERICAL_NP_ARRAY_TEST, clf, 3, 'regressor', treat_as_categorical=True)
    assert np.allclose(ice, NUMERICAL_NP_ICE_CAT_REGRESSION)
    assert np.allclose(linespace, NUMERICAL_NP_LINESPACE_CAT)

    ice, linespace = ftmfi.individual_conditional_expectation(
        NUMERICAL_NP_ARRAY_TEST, clf, 3, 'regressor')
    assert np.allclose(ice, NUMERICAL_NP_ICE_100_REGRESSION)
    assert np.allclose(linespace, NUMERICAL_NP_LINESPACE_100)

    clf = fum.KNN(k=2, mode='regressor')
    clf.fit(NUMERICAL_STRUCT_ARRAY, 
            NUMERICAL_NP_ARRAY_TARGET.astype(np.float32))

    ice, linespace = ftmfi.individual_conditional_expectation(
        NUMERICAL_STRUCT_ARRAY_TEST, clf, 'd', 'regressor', steps_number=3)
    assert np.allclose(ice, NUMERICAL_NP_ICE_REGRESSION)
    assert np.allclose(linespace, NUMERICAL_NP_LINESPACE)

    ice, linespace = ftmfi.individual_conditional_expectation(
        NUMERICAL_STRUCT_ARRAY_TEST, clf, 'd', 'regressor',
        treat_as_categorical=True)
    assert np.allclose(ice, NUMERICAL_NP_ICE_CAT_REGRESSION)
    assert np.allclose(linespace, NUMERICAL_NP_LINESPACE_CAT)

    ice, linespace = ftmfi.individual_conditional_expectation(
        NUMERICAL_STRUCT_ARRAY_TEST, clf, 'd', 'regressor')
    assert np.allclose(ice, NUMERICAL_NP_ICE_100_REGRESSION)
    assert np.allclose(linespace, NUMERICAL_NP_LINESPACE_100)

    clf = fum.KNN(k=2, mode='regressor')
    clf.fit(CATEGORICAL_NP_ARRAY, 
            CATEGORICAL_NP_ARRAY_TARGET.astype(np.float32))
    
    ice, linespace = ftmfi.individual_conditional_expectation(
        CATEGORICAL_NP_ARRAY_TEST, clf, 2, 'regressor')
    assert np.allclose(ice, CATEGORICAL_NP_ICE_REGRESSION)
    assert np.array_equal(linespace, CATEGORICAL_NP_LINESPACE)

    clf = fum.KNN(k=2, mode='regressor')
    clf.fit(CATEGORICAL_STRUCT_ARRAY, 
            CATEGORICAL_NP_ARRAY_TARGET.astype(np.float32))

    ice, linespace = ftmfi.individual_conditional_expectation(
        CATEGORICAL_STRUCT_ARRAY_TEST, clf, 'c', 'regressor')
    assert np.allclose(ice, CATEGORICAL_NP_ICE_REGRESSION)
    assert np.array_equal(linespace, CATEGORICAL_NP_LINESPACE)

    clf = fum.KNN(k=2, mode='regressor')
    clf.fit(MIXED_ARRAY, 
            MIXED_ARRAY_TARGET_REGRESSION)

    ice, linespace = ftmfi.individual_conditional_expectation(
        MIXED_ARRAY_TEST, clf, 'a', 'regressor', steps_number=3, 
        exclude_rows=1)
    assert np.allclose(ice, MIXED_ICE_NUMERICAL_REGRESSION)
    assert np.array_equal(linespace, MIXED_LINESPACE_NUMERICAL)

    ice, linespace = ftmfi.individual_conditional_expectation(
        MIXED_ARRAY_TEST, clf, 'b', 'regressor', exclude_rows=1)
    assert np.allclose(ice, MIXED_ICE_CATEGORICAL_REGRESSION)
    assert np.array_equal(linespace, MIXED_LINESPACE_CATEGORICAL)

    ###########################################################################
    # 2-D INDIVIDUAL CONDITIONAL EXPECTATION
    # Classical arrays
    clf = fum.KNN(k=2)
    clf.fit(NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY_TARGET)

    ice, linespace = ftmfi.individual_conditional_expectation(
        NUMERICAL_NP_ARRAY_TEST, clf, [0 ,3], 'classifier',
        steps_number=[3, 3])
    assert np.allclose(ice, NUMERICAL_NP_ICE_2D)
    for line, correct_line in zip(linespace, NUMERICAL_NP_LINESPACE_2D):
        assert np.allclose(line, correct_line)
    
    with pytest.warns(UserWarning) as warning:
        ice, linespace = ftmfi.individual_conditional_expectation(
            NUMERICAL_NP_ARRAY_TEST, clf, [0 ,3], 'classifier',
            steps_number=[3, 3], treat_as_categorical=[False, True])
    assert len(warning) == 1
    assert str(warning[0].message) == steps_n_warning
    assert np.allclose(ice, NUMERICAL_NP_ICE_CAT_2D)
    for line, correct_line in zip(linespace, NUMERICAL_NP_LINESPACE_CAT_2D):
        assert np.allclose(line, correct_line)

    ice, linespace = ftmfi.individual_conditional_expectation(
        NUMERICAL_NP_ARRAY_TEST, clf, [0, 3], 'classifier',
        steps_number=[3, None])
    assert np.allclose(ice, NUMERICAL_NP_ICE_100_2D)
    for line, correct_line in zip(linespace, NUMERICAL_NP_LINESPACE_100_2D):
        assert np.allclose(line, correct_line)

    clf = fum.KNN(k=2)
    clf.fit(CATEGORICAL_NP_ARRAY, CATEGORICAL_NP_ARRAY_TARGET)

    ice, linespace = ftmfi.individual_conditional_expectation(
        CATEGORICAL_NP_ARRAY_TEST, clf, [0 ,2], 'classifier')
    assert np.allclose(ice, CATEGORICAL_NP_ICE_2D)
    for line, correct_line in zip(linespace, CATEGORICAL_NP_LINESPACE_2D):
        assert np.array_equal(line, correct_line)

    with pytest.warns(UserWarning) as warning:
        ice, linespace = ftmfi.individual_conditional_expectation(
        CATEGORICAL_NP_ARRAY_TEST, clf, [0 ,2], 'classifier',
        steps_number=[None, 42])
    assert len(warning) == 1
    assert str(warning[0].message) == steps_n_warning
    assert np.allclose(ice, CATEGORICAL_NP_ICE_2D)
    for line, correct_line in zip(linespace, CATEGORICAL_NP_LINESPACE_2D):
        assert np.array_equal(line, correct_line)


    # Structured Data
    clf = fum.KNN(k=2, mode='classifier')
    clf.fit(NUMERICAL_STRUCT_ARRAY, 
            NUMERICAL_NP_ARRAY_TARGET.astype(np.float32))

    ice, linespace = ftmfi.individual_conditional_expectation(
        NUMERICAL_STRUCT_ARRAY_TEST, clf, ['a', 'd'], 'classifier',
        steps_number=3)
    assert np.allclose(ice, NUMERICAL_NP_ICE_2D)
    for line, correct_line in zip(linespace, NUMERICAL_NP_LINESPACE_2D):
        assert np.allclose(line, correct_line)

    ice, linespace = ftmfi.individual_conditional_expectation(
        NUMERICAL_STRUCT_ARRAY_TEST, clf, ['a', 'd'], 'classifier',
        steps_number=[3, None], treat_as_categorical=[False, True])
    assert np.allclose(ice, NUMERICAL_NP_ICE_CAT_2D)
    for line, correct_line in zip(linespace, NUMERICAL_NP_LINESPACE_CAT_2D):
        assert np.allclose(line, correct_line)

    clf = fum.KNN(k=2)
    clf.fit(CATEGORICAL_STRUCT_ARRAY, CATEGORICAL_NP_ARRAY_TARGET)

    ice, linespace = ftmfi.individual_conditional_expectation(
        CATEGORICAL_STRUCT_ARRAY_TEST, clf, ['a', 'c'], 'classifier')
    assert np.allclose(ice, CATEGORICAL_NP_ICE_2D)
    for line, correct_line in zip(linespace, CATEGORICAL_NP_LINESPACE_2D):
        assert np.array_equal(line, correct_line)

    with pytest.warns(UserWarning) as warning:
        ice, linespace = ftmfi.individual_conditional_expectation(
        CATEGORICAL_STRUCT_ARRAY_TEST, clf, ['a', 'c'], 'classifier',
        steps_number=[None, 42])
    assert len(warning) == 1
    assert str(warning[0].message) == steps_n_warning
    assert np.allclose(ice, CATEGORICAL_NP_ICE_2D)
    for line, correct_line in zip(linespace, CATEGORICAL_NP_LINESPACE_2D):
        assert np.array_equal(line, correct_line)

    # Mixed arrays
    clf = fum.KNN(k=2, mode='classifier')
    clf.fit(MIXED_ARRAY, MIXED_ARRAY_TARGET)

    ice, linespace = ftmfi.individual_conditional_expectation(
        MIXED_ARRAY_TEST, clf, ['a', 'c'], 'classifier', steps_number=3,
        exclude_rows=1)
    assert np.allclose(ice, MIXED_ICE_NUMERICAL_2D)
    for line, correct_line in zip(linespace, MIXED_LINESPACE_NUMERICAL_2D):
        assert np.allclose(line, correct_line)
    
    ice, linespace = ftmfi.individual_conditional_expectation(
        MIXED_ARRAY_TEST, clf, ['b', 'd'], 'classifier')
    assert np.allclose(ice, MIXED_ICE_CATEGORICAL_2D)
    for line, correct_line in zip(linespace, MIXED_LINESPACE_CATEGORICAL_2D):
        assert np.array_equal(line, correct_line)

    ice, linespace = ftmfi.individual_conditional_expectation(
        MIXED_ARRAY_TEST, clf, ['a', 'b'], 'classifier',
        steps_number=[3, None])
    assert np.allclose(ice, MIXED_ICE_MIX_2D)
    for line, correct_line in zip(linespace, MIXED_LINESPACE_MIX_2D):
        assert np.array_equal(line, correct_line)


def test_partial_dependence_ice():
    """
    Tests Partial Dependence calculations from an ICE array.

    Tests :func:`fatf.transparency.models.feature_influence.
    partial_dependence_ice` function.
    """
    value_error_structured = 'The ice_array should not be structured.'
    value_error_not_numerical = 'The ice_array should be purely numerical.'
    incorrect_shape_error = ('The ice_array should be 3-dimensional or '
                             '4-dimensional for 2 feature ICE.')

    with pytest.raises(ValueError) as exin:
        ftmfi.partial_dependence_ice(np.array([(1, )], dtype=[('a', int)]))
    assert str(exin.value) == value_error_structured

    with pytest.raises(ValueError) as exin:
        ftmfi.partial_dependence_ice(np.array([[1, 'a', 2]]))
    assert str(exin.value) == value_error_not_numerical

    with pytest.raises(IncorrectShapeError) as exin:
        ftmfi.partial_dependence_ice(ONE_D_ARRAY)
    assert str(exin.value) == incorrect_shape_error

    # Test PD
    pd, var = ftmfi.partial_dependence_ice(NUMERICAL_NP_ICE)
    assert np.array_equal(pd, NUMERICAL_NP_PD)
    assert np.array_equal(var, NUMERICAL_NP_VARIANCE)

    pd, var = ftmfi.partial_dependence_ice(NUMERICAL_NP_ICE_CAT)
    assert np.array_equal(pd, NUMERICAL_NP_PD_CAT)
    assert np.array_equal(var, NUMERICAL_NP_VARIANCE_CAT)

    pd, var = ftmfi.partial_dependence_ice(NUMERICAL_NP_ICE_100)
    assert np.array_equal(pd, NUMERICAL_NP_PD_100)
    assert np.array_equal(var, NUMERICAL_NP_VARIANCE_100)

    pd, var = ftmfi.partial_dependence_ice(CATEGORICAL_NP_ICE)
    assert np.array_equal(pd, CATEGORICAL_NP_PD)
    assert np.array_equal(var, CATEGORICAL_NP_VARIANCE)

    pd, var = ftmfi.partial_dependence_ice(MIXED_ICE_NUMERICAL)
    assert np.array_equal(pd, MIXED_PD_NUMERICAL)
    assert np.array_equal(var, MIXED_VARIANCE_NUMERICAL)

    pd, var = ftmfi.partial_dependence_ice(MIXED_ICE_CATEGORICAL)
    assert np.array_equal(pd, MIXED_PD_CATEGORICAL)
    assert np.array_equal(var, MIXED_VARIANCE_CATEGORICAL)

    # Test row exclusion
    pd, var = ftmfi.partial_dependence_ice(MIXED_ICE_CATEGORICAL, include_rows=0)
    assert np.array_equal(pd, MIXED_ICE_CATEGORICAL[0])

    assert np.array_equal(var, np.array([[0., 0., 0.], [0., 0., 0.]]))

    pd, var = ftmfi.partial_dependence_ice(MIXED_ICE_CATEGORICAL, include_rows=[0])
    assert np.array_equal(pd, MIXED_ICE_CATEGORICAL[0])
    assert np.array_equal(var, np.array([[0., 0., 0.], [0., 0., 0.]]))

    pd, var = ftmfi.partial_dependence_ice(MIXED_ICE_CATEGORICAL, exclude_rows=1)
    assert np.array_equal(pd, MIXED_ICE_CATEGORICAL[0])
    assert np.array_equal(var, np.array([[0., 0., 0.], [0., 0., 0.]]))

    pd, var = ftmfi.partial_dependence_ice(
        MIXED_ICE_CATEGORICAL, exclude_rows=[1])
    assert np.array_equal(pd, MIXED_ICE_CATEGORICAL[0])
    assert np.array_equal(var, np.array([[0., 0., 0.], [0., 0., 0.]]))

    pd, var = ftmfi.partial_dependence_ice(
        MIXED_ICE_CATEGORICAL, include_rows=[1, 0], exclude_rows=[1])
    assert np.array_equal(pd, MIXED_ICE_CATEGORICAL[0])
    assert np.array_equal(var, np.array([[0., 0., 0.], [0., 0., 0.]]))

    # 2-D PD
    pd, var = ftmfi.partial_dependence_ice(NUMERICAL_NP_ICE_2D)
    assert np.array_equal(pd, NUMERICAL_NP_PD_2D)

    pd, var = ftmfi.partial_dependence_ice(NUMERICAL_NP_ICE_CAT_2D)
    assert np.array_equal(pd, NUMERICAL_NP_PD_CAT_2D)

    pd, var = ftmfi.partial_dependence_ice(NUMERICAL_NP_ICE_100_2D)
    assert np.array_equal(pd, NUMERICAL_NP_PD_100_2D)

    pd, var = ftmfi.partial_dependence_ice(CATEGORICAL_NP_ICE_2D)
    assert np.array_equal(pd, CATEGORICAL_NP_PD_2D)

    pd, var = ftmfi.partial_dependence_ice(MIXED_ICE_NUMERICAL_2D)
    assert np.array_equal(pd, MIXED_PD_NUMERICAL_2D)

    pd, var = ftmfi.partial_dependence_ice(MIXED_ICE_CATEGORICAL_2D)
    assert np.allclose(pd, MIXED_PD_CATEGORICAL_2D, atol=1e-2)


def test_partial_dependence():
    """
    Tests Partial Dependence calculations.

    Tests :func:`fatf.transparency.models.feature_influence.
    partial_dependence` function.
    """
    clf = fum.KNN(k=2)
    clf.fit(NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY_TARGET)
    clf_struct = fum.KNN(k=2)
    clf_struct.fit(NUMERICAL_STRUCT_ARRAY, NUMERICAL_NP_ARRAY_TARGET)

    # Test PD
    pd, linespace, var = ftmfi.partial_dependence(
        NUMERICAL_STRUCT_ARRAY_TEST,
        clf_struct,
        'd',
        treat_as_categorical=False,
        steps_number=3)
    assert np.allclose(pd, NUMERICAL_NP_PD)
    assert np.allclose(linespace, NUMERICAL_NP_LINESPACE)
    assert np.array_equal(var, NUMERICAL_NP_VARIANCE)

    pd, linespace, var = ftmfi.partial_dependence(
        NUMERICAL_NP_ARRAY_TEST, clf, 3, treat_as_categorical=False)
    assert np.allclose(pd, NUMERICAL_NP_PD_100)
    assert np.allclose(linespace, NUMERICAL_NP_LINESPACE_100)
    assert np.array_equal(var, NUMERICAL_NP_VARIANCE_100)

    pd, linespace, var = ftmfi.partial_dependence(
        NUMERICAL_STRUCT_ARRAY_TEST,
        clf_struct,
        'd',
        treat_as_categorical=True)
    assert np.allclose(pd, NUMERICAL_NP_PD_CAT)
    assert np.allclose(linespace, NUMERICAL_NP_LINESPACE_CAT)
    assert np.array_equal(var, NUMERICAL_NP_VARIANCE_CAT)

    pd, linespace, var = ftmfi.partial_dependence(
        NUMERICAL_NP_ARRAY_TEST, clf, 3, treat_as_categorical=True)
    assert np.allclose(pd, NUMERICAL_NP_PD_CAT)
    assert np.allclose(linespace, NUMERICAL_NP_LINESPACE_CAT)
    assert np.array_equal(var, NUMERICAL_NP_VARIANCE_CAT)

    clf = fum.KNN(k=2)
    clf.fit(CATEGORICAL_NP_ARRAY, CATEGORICAL_NP_ARRAY_TARGET)
    clf_struct = fum.KNN(k=2)
    clf_struct.fit(CATEGORICAL_STRUCT_ARRAY, CATEGORICAL_NP_ARRAY_TARGET)

    pd, linespace, var = ftmfi.partial_dependence(
        CATEGORICAL_NP_ARRAY_TEST, clf, 2, treat_as_categorical=True)
    assert np.allclose(pd, CATEGORICAL_NP_PD)
    assert np.array_equal(linespace, CATEGORICAL_NP_LINESPACE)
    assert np.array_equal(var, CATEGORICAL_NP_VARIANCE)

    pd, linespace, VAR = ftmfi.partial_dependence(
        CATEGORICAL_STRUCT_ARRAY_TEST,
        clf_struct,
        'c',
        treat_as_categorical=True)
    assert np.allclose(pd, CATEGORICAL_NP_PD)
    assert np.array_equal(linespace, CATEGORICAL_NP_LINESPACE)
    assert np.array_equal(var, CATEGORICAL_NP_VARIANCE)

    # Test row exclusion on a mixed array
    clf = fum.KNN(k=2)
    clf.fit(MIXED_ARRAY, MIXED_ARRAY_TARGET)

    pd, linespace, var = ftmfi.partial_dependence(
        MIXED_ARRAY_TEST,
        clf,
        'a',
        steps_number=3,
        include_rows=[0, 2],
        exclude_rows=1)
    assert np.allclose(pd, MIXED_PD_NUMERICAL)
    assert np.array_equal(linespace, MIXED_LINESPACE_NUMERICAL)
    assert np.array_equal(var, MIXED_VARIANCE_NUMERICAL)

    pd, linespace, var = ftmfi.partial_dependence(
        MIXED_ARRAY_TEST, clf, 'b', exclude_rows=1)
    assert np.allclose(pd, MIXED_PD_CATEGORICAL)
    assert np.array_equal(linespace, MIXED_LINESPACE_CATEGORICAL)
    assert np.array_equal(var, MIXED_VARIANCE_CATEGORICAL)

     # Test Regression
    clf = fum.KNN(k=2, mode='regressor')
    clf.fit(NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY_TARGET.astype(np.float32))

    pd, linespace, var = ftmfi.partial_dependence(
        NUMERICAL_NP_ARRAY_TEST, clf, 3, 'regressor', steps_number=3)
    assert np.allclose(pd, NUMERICAL_NP_PD_REGRESSION)
    assert np.allclose(linespace, NUMERICAL_NP_LINESPACE)
    assert np.allclose(var, NUMERICAL_NP_VARIANCE_REGRESSION)

    pd, linespace, var = ftmfi.partial_dependence(
        NUMERICAL_NP_ARRAY_TEST, clf, 3, 'regressor', treat_as_categorical=True)
    assert np.allclose(pd, NUMERICAL_NP_PD_CAT_REGRESSION)
    assert np.allclose(linespace, NUMERICAL_NP_LINESPACE_CAT)
    assert np.allclose(var, NUMERICAL_NP_VARIANCE_CAT_REGRESSION)

    pd, linespace, var = ftmfi.partial_dependence(
        NUMERICAL_NP_ARRAY_TEST, clf, 3, 'regressor')
    assert np.allclose(pd, NUMERICAL_NP_PD_100_REGRESSION)
    assert np.allclose(linespace, NUMERICAL_NP_LINESPACE_100)
    assert np.allclose(var, NUMERICAL_NP_VARIANCE_100_REGRESSION)

    clf = fum.KNN(k=2, mode='regressor')
    clf.fit(NUMERICAL_STRUCT_ARRAY, 
            NUMERICAL_NP_ARRAY_TARGET.astype(np.float32))

    pd, linespace, var = ftmfi.partial_dependence(
        NUMERICAL_STRUCT_ARRAY_TEST, clf, 'd', 'regressor', steps_number=3)
    assert np.allclose(pd, NUMERICAL_NP_PD_REGRESSION)
    assert np.allclose(linespace, NUMERICAL_NP_LINESPACE)
    assert np.allclose(var, NUMERICAL_NP_VARIANCE_REGRESSION)

    pd, linespace, var = ftmfi.partial_dependence(
        NUMERICAL_STRUCT_ARRAY_TEST, clf, 'd', 'regressor',
        treat_as_categorical=True)
    assert np.allclose(pd, NUMERICAL_NP_PD_CAT_REGRESSION)
    assert np.allclose(linespace, NUMERICAL_NP_LINESPACE_CAT)
    assert np.allclose(var, NUMERICAL_NP_VARIANCE_CAT_REGRESSION)

    pd, linespace, var = ftmfi.partial_dependence(
        NUMERICAL_STRUCT_ARRAY_TEST, clf, 'd', 'regressor')
    assert np.allclose(pd, NUMERICAL_NP_PD_100_REGRESSION)
    assert np.allclose(linespace, NUMERICAL_NP_LINESPACE_100)
    assert np.allclose(var, NUMERICAL_NP_VARIANCE_100_REGRESSION)

    clf = fum.KNN(k=2, mode='regressor')
    clf.fit(CATEGORICAL_NP_ARRAY, 
            CATEGORICAL_NP_ARRAY_TARGET.astype(np.float32))
    
    pd, linespace, var = ftmfi.partial_dependence(
        CATEGORICAL_NP_ARRAY_TEST, clf, 2, 'regressor')
    assert np.allclose(pd, CATEGORICAL_NP_PD_REGRESSION)
    assert np.array_equal(linespace, CATEGORICAL_NP_LINESPACE)
    assert np.allclose(var, CATEGORICAL_NP_VARIANCE_REGRESSION)

    clf = fum.KNN(k=2, mode='regressor')
    clf.fit(CATEGORICAL_STRUCT_ARRAY, 
            CATEGORICAL_NP_ARRAY_TARGET.astype(np.float32))

    pd, linespace, var = ftmfi.partial_dependence(
        CATEGORICAL_STRUCT_ARRAY_TEST, clf, 'c', 'regressor')
    assert np.allclose(pd, CATEGORICAL_NP_PD_REGRESSION)
    assert np.array_equal(linespace, CATEGORICAL_NP_LINESPACE)
    assert np.allclose(var, CATEGORICAL_NP_VARIANCE_REGRESSION)

    clf = fum.KNN(k=2, mode='regressor')
    clf.fit(MIXED_ARRAY, 
            MIXED_ARRAY_TARGET_REGRESSION)

    pd, linespace, var = ftmfi.partial_dependence(
        MIXED_ARRAY_TEST, clf, 'a', 'regressor', steps_number=3, 
        exclude_rows=1)
    assert np.allclose(pd, MIXED_PD_NUMERICAL_REGRESSION)
    assert np.array_equal(linespace, MIXED_LINESPACE_NUMERICAL)
    assert np.array_equal(var, MIXED_VARIANCE_NUMERICAL_REGRESSION)

    pd, linespace, var = ftmfi.partial_dependence(
        MIXED_ARRAY_TEST, clf, 'b', 'regressor', exclude_rows=1)
    assert np.allclose(pd, MIXED_PD_CATEGORICAL_REGRESSION)
    assert np.array_equal(linespace, MIXED_LINESPACE_CATEGORICAL)

    clf = fum.KNN(k=2, mode='classifier')
    clf.fit(NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY_TARGET)

    # Test 2-D Partial Dependence
    pd, linespace, var = ftmfi.partial_dependence(
        NUMERICAL_NP_ARRAY_TEST, clf, [0, 3], 'classifier',
        steps_number=[3, 3], treat_as_categorical=[False, False])
    assert np.allclose(pd, NUMERICAL_NP_PD_2D)
    for line, correct_line in zip(linespace, NUMERICAL_NP_LINESPACE_2D):
        assert np.allclose(line, correct_line)

    pd, linespace, var = ftmfi.partial_dependence(
        NUMERICAL_NP_ARRAY_TEST, clf, [0, 3], 'classifier',
        steps_number=[3, None], treat_as_categorical=[False, True])
    assert np.allclose(pd, NUMERICAL_NP_PD_CAT_2D)
    for line, correct_line in zip(linespace, NUMERICAL_NP_LINESPACE_CAT_2D):
        assert np.allclose(line, correct_line)

    pd, linespace, var = ftmfi.partial_dependence(
        NUMERICAL_NP_ARRAY_TEST, clf, [0, 3], 'classifier',
        steps_number=[3, None], treat_as_categorical=[False, False])
    assert np.allclose(pd, NUMERICAL_NP_PD_100_2D)
    for line, correct_line in zip(linespace, NUMERICAL_NP_LINESPACE_100_2D):
        assert np.allclose(line, correct_line)
    
    clf = fum.KNN(k=2, mode='classifier')
    clf.fit(NUMERICAL_STRUCT_ARRAY, NUMERICAL_NP_ARRAY_TARGET)

    pd, linespace, var = ftmfi.partial_dependence(
        NUMERICAL_STRUCT_ARRAY_TEST, clf, ['a', 'd'], 'classifier',
        steps_number=[3, 3], treat_as_categorical=[False, False])
    assert np.allclose(pd, NUMERICAL_NP_PD_2D)
    for line, correct_line in zip(linespace, NUMERICAL_NP_LINESPACE_2D):
        assert np.allclose(line, correct_line)

    pd, linespace, var = ftmfi.partial_dependence(
        NUMERICAL_STRUCT_ARRAY_TEST, clf, ['a', 'd'], 'classifier',
        steps_number=[3, None], treat_as_categorical=[False, True])
    assert np.allclose(pd, NUMERICAL_NP_PD_CAT_2D)
    for line, correct_line in zip(linespace, NUMERICAL_NP_LINESPACE_CAT_2D):
        assert np.allclose(line, correct_line)

    pd, linespace, var = ftmfi.partial_dependence(
        NUMERICAL_STRUCT_ARRAY_TEST, clf, ['a', 'd'], 'classifier',
        steps_number=[3, None], treat_as_categorical=[False, False])
    assert np.allclose(pd, NUMERICAL_NP_PD_100_2D)
    for line, correct_line in zip(linespace, NUMERICAL_NP_LINESPACE_100_2D):
        assert np.allclose(line, correct_line)

    clf = fum.KNN(k=2, mode='classifier')
    clf.fit(CATEGORICAL_NP_ARRAY, CATEGORICAL_NP_ARRAY_TARGET)

    pd, linespace, var = ftmfi.partial_dependence(
        CATEGORICAL_NP_ARRAY_TEST, clf, [0, 2], 'classifier',
        steps_number=[None, None])
    assert np.allclose(pd, CATEGORICAL_NP_PD_2D)
    assert np.array_equal(linespace, CATEGORICAL_NP_LINESPACE_2D)


    clf = fum.KNN(k=2, mode='classifier')
    clf.fit(CATEGORICAL_STRUCT_ARRAY, CATEGORICAL_NP_ARRAY_TARGET)

    pd, linespace, var = ftmfi.partial_dependence(
        CATEGORICAL_STRUCT_ARRAY_TEST, clf, ['a', 'c'], 'classifier')
    assert np.allclose(pd, CATEGORICAL_NP_PD_2D)
    assert np.array_equal(linespace, CATEGORICAL_NP_LINESPACE_2D)

    clf = fum.KNN(k=2, mode='classifier')
    clf.fit(MIXED_ARRAY, MIXED_ARRAY_TARGET)

    ice, linespace, var = ftmfi.partial_dependence(
        MIXED_ARRAY_TEST, clf, ['a', 'c'], 'classifier', steps_number=3,
        exclude_rows=1)
    assert np.allclose(ice, MIXED_PD_NUMERICAL_2D)
    for line, correct_line in zip(linespace, MIXED_LINESPACE_NUMERICAL_2D):
        assert np.allclose(line, correct_line)
    
    ice, linespace, var = ftmfi.partial_dependence(
        MIXED_ARRAY_TEST, clf, ['b', 'd'], 'classifier')
    assert np.allclose(ice, MIXED_PD_CATEGORICAL_2D, atol=1e-2)
    for line, correct_line in zip(linespace, MIXED_LINESPACE_CATEGORICAL_2D):
        assert np.array_equal(line, correct_line)

    ice, linespace, var = ftmfi.partial_dependence(
        MIXED_ARRAY_TEST, clf, ['a', 'b'], 'classifier',
        steps_number=[3, None])
    assert np.allclose(ice, MIXED_PD_MIX_2D, atol=1e-2)
    for line, correct_line in zip(linespace, MIXED_LINESPACE_MIX_2D):
        assert np.array_equal(line, correct_line)
