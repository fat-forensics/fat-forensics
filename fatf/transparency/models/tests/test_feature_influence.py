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
NUMERICAL_NP_PD = np.array([
    [0.50, 0.0, 0.50],
    [0.75, 0.0, 0.25],
    [0.75, 0.0, 0.25]])
NUMERICAL_NP_ICE_CAT = np.array([
    [[1., 0., 0.],
     [1., 0., 0.]],
    [[0.0, 0., 1.0],
     [0.5, 0., 0.5]]])
NUMERICAL_NP_PD_CAT = np.array([
    [0.50, 0.0, 0.50],
    [0.75, 0.0, 0.25]])
NUMERICAL_NP_ICE_100 = np.array(
    [100 * [[1.0, 0.0, 0.0]],
     46 * [[0.0, 0.0, 1.0]] + 54 * [[0.5, 0.0, 0.5]]])
NUMERICAL_NP_PD_100 = np.array(
    46 * [[0.5, 0.0, 0.5]] + 54 * [[0.75, 0.00, 0.25]])
NUMERICAL_NP_LINESPACE = np.array([0.32, 0.41, 0.5])
NUMERICAL_NP_LINESPACE_CAT = np.array([0.32, 0.5])
NUMERICAL_NP_LINESPACE_100 = np.linspace(0.32, 0.5, 100)

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
CATEGORICAL_NP_PD = np.array([
    [0.25, 0.75],
    [0.25, 0.75]])
CATEGORICAL_NP_LINESPACE = np.array(['c', 'g'])

MIXED_ARRAY_TEST = np.array(
    [(0, 'a', 0.08, 'a'),
     (1, 'a', 0.88, 'bb'),
     (1, 'f', 0.07, 'bb')],
    dtype=[('a', 'i'), ('b', 'U1'), ('c', 'f'), ('d', 'U2')])
MIXED_ARRAY_TARGET = np.array(['a', 'b', 'c', 'a', 'b', 'c'])
MIXED_ICE_NUMERICAL = np.array([
    [[1.0, 0.0, 0.0],
     [1.0, 0.0, 0.0],
     [1.0, 0.0, 0.0]],
    [[0.0, 0.5, 0.5],
     [0.0, 0.5, 0.5],
     [0.0, 0.5, 0.5]]])
MIXED_PD_NUMERICAL = np.array([
    [0.5, 0.25, 0.25],
    [0.5, 0.25, 0.25],
    [0.5, 0.25, 0.25]])
MIXED_LINESPACE_NUMERICAL = np.array([0, 0.5, 1])
MIXED_ICE_CATEGORICAL = np.array([
    [[1.0, 0.0, 0.0],
     [0.5, 0.5, 0.0]],
    [[0.5, 0.0, 0.5],
     [0.0, 0.5, 0.5]]])
MIXED_PD_CATEGORICAL = np.array([
    [0.75, 0.0, 0.25],
    [0.25, 0.5, 0.25]])
MIXED_LINESPACE_CATEGORICAL = np.array(['a', 'f'])
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


def test_is_valid_input():
    """
    Tests :func:`fatf.transparency.models.feature_influence._is_valid_input`.
    """
    knn_model = fum.KNN()

    # Data
    msg = 'The input dataset must be a 2-dimensional array.'
    with pytest.raises(IncorrectShapeError) as exin:
        ftmfi._input_is_valid(ONE_D_ARRAY, None, None, None, None)
    assert str(exin.value) == msg

    msg = ('The input dataset must only contain base types (textual and '
           'numerical).')
    with pytest.raises(ValueError) as exin:
        ftmfi._input_is_valid(NOT_BASE_NP_ARRAY, None, None, None, None)
    assert str(exin.value) == msg

    # Model
    msg = ('This functionality requires the model to be capable of outputting '
           'probabilities via predict_proba method.')
    model = InvalidModel()
    with pytest.warns(UserWarning) as warning:
        with pytest.raises(IncompatibleModelError) as exin:
            ftmfi._input_is_valid(BASE_STRUCTURED_ARRAY, model, None, None,
                                  None)
        assert str(exin.value) == msg
    assert len(warning) == 1
    msg = ('Model object characteristics are neither consistent with '
           'supervised nor unsupervised models.\n\n'
           '--> Unsupervised models <--\n'
           "The 'fit' method of the *InvalidModel* (model) class has "
           'incorrect number (2) of the required parameters. It needs to have '
           'exactly 1 required parameter(s). Try using optional parameters if '
           'you require more functionality.\n\n'
           '--> Supervised models <--\n'
           "The *InvalidModel* (model) class is missing 'predict_proba' "
           'method.')
    assert str(warning[0].message) == msg

    # Feature index
    msg = 'Provided feature index is not valid for the input dataset.'
    with pytest.raises(IndexError) as exin:
        ftmfi._input_is_valid(BASE_STRUCTURED_ARRAY, knn_model, 0, None, None)
    assert str(exin.value) == msg
    with pytest.raises(IndexError) as exin:
        ftmfi._input_is_valid(BASE_NP_ARRAY, knn_model, 'numerical', None,
                              None)
    assert str(exin.value) == msg

    # Steps number
    msg = 'steps_number parameter has to either be None or an integer.'
    with pytest.raises(TypeError) as exin:
        ftmfi._input_is_valid(BASE_NP_ARRAY, knn_model, 1, None, 'a')
    assert str(exin.value) == msg

    msg = 'steps_number has to be at least 2.'
    with pytest.raises(ValueError) as exin:
        ftmfi._input_is_valid(BASE_NP_ARRAY, knn_model, 1, None, 1)
    assert str(exin.value) == msg

    # Treat as categorical
    msg = 'treat_as_categorical has to either be None or a boolean.'
    with pytest.raises(TypeError) as exin:
        ftmfi._input_is_valid(BASE_NP_ARRAY, knn_model, 1, 'a', None)
    assert str(exin.value) == msg

    # Functional
    assert ftmfi._input_is_valid(BASE_NP_ARRAY, knn_model, 1, None, 2)
    assert ftmfi._input_is_valid(BASE_NP_ARRAY, knn_model, 1, False, 5)
    # Steps number will be ignored anyway
    assert ftmfi._input_is_valid(BASE_NP_ARRAY, knn_model, 1, True, 2)


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
    incorrect_shape_3d = 'The ice_array should be 3-dimensional.'
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


def test_partial_dependence_ice():
    """
    Tests Partial Dependence calculations from an ICE array.

    Tests :func:`fatf.transparency.models.feature_influence.
    partial_dependence_ice` function.
    """
    value_error_structured = 'The ice_array should not be structured.'
    value_error_not_numerical = 'The ice_array should be purely numerical.'
    incorrect_shape_error = 'The ice_array should be 3-dimensional.'

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
    pd = ftmfi.partial_dependence_ice(NUMERICAL_NP_ICE)
    assert np.array_equal(pd, NUMERICAL_NP_PD)

    pd = ftmfi.partial_dependence_ice(NUMERICAL_NP_ICE_CAT)
    assert np.array_equal(pd, NUMERICAL_NP_PD_CAT)

    pd = ftmfi.partial_dependence_ice(NUMERICAL_NP_ICE_100)
    assert np.array_equal(pd, NUMERICAL_NP_PD_100)

    pd = ftmfi.partial_dependence_ice(CATEGORICAL_NP_ICE)
    assert np.array_equal(pd, CATEGORICAL_NP_PD)

    pd = ftmfi.partial_dependence_ice(MIXED_ICE_NUMERICAL)
    assert np.array_equal(pd, MIXED_PD_NUMERICAL)

    pd = ftmfi.partial_dependence_ice(MIXED_ICE_CATEGORICAL)
    assert np.array_equal(pd, MIXED_PD_CATEGORICAL)

    # Test row exclusion
    pd = ftmfi.partial_dependence_ice(MIXED_ICE_CATEGORICAL, include_rows=0)
    assert np.array_equal(pd, MIXED_ICE_CATEGORICAL[0])

    pd = ftmfi.partial_dependence_ice(MIXED_ICE_CATEGORICAL, include_rows=[0])
    assert np.array_equal(pd, MIXED_ICE_CATEGORICAL[0])

    pd = ftmfi.partial_dependence_ice(MIXED_ICE_CATEGORICAL, exclude_rows=1)
    assert np.array_equal(pd, MIXED_ICE_CATEGORICAL[0])

    pd = ftmfi.partial_dependence_ice(MIXED_ICE_CATEGORICAL, exclude_rows=[1])
    assert np.array_equal(pd, MIXED_ICE_CATEGORICAL[0])

    pd = ftmfi.partial_dependence_ice(
        MIXED_ICE_CATEGORICAL, include_rows=[1, 0], exclude_rows=[1])
    assert np.array_equal(pd, MIXED_ICE_CATEGORICAL[0])


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
    pd, linespace = ftmfi.partial_dependence(
        NUMERICAL_STRUCT_ARRAY_TEST,
        clf_struct,
        'd',
        treat_as_categorical=False,
        steps_number=3)
    assert np.allclose(pd, NUMERICAL_NP_PD)
    assert np.allclose(linespace, NUMERICAL_NP_LINESPACE)

    pd, linespace = ftmfi.partial_dependence(
        NUMERICAL_NP_ARRAY_TEST, clf, 3, treat_as_categorical=False)
    assert np.allclose(pd, NUMERICAL_NP_PD_100)
    assert np.allclose(linespace, NUMERICAL_NP_LINESPACE_100)

    pd, linespace = ftmfi.partial_dependence(
        NUMERICAL_STRUCT_ARRAY_TEST,
        clf_struct,
        'd',
        treat_as_categorical=True)
    assert np.allclose(pd, NUMERICAL_NP_PD_CAT)
    assert np.allclose(linespace, NUMERICAL_NP_LINESPACE_CAT)

    pd, linespace = ftmfi.partial_dependence(
        NUMERICAL_NP_ARRAY_TEST, clf, 3, treat_as_categorical=True)
    assert np.allclose(pd, NUMERICAL_NP_PD_CAT)
    assert np.allclose(linespace, NUMERICAL_NP_LINESPACE_CAT)

    clf = fum.KNN(k=2)
    clf.fit(CATEGORICAL_NP_ARRAY, CATEGORICAL_NP_ARRAY_TARGET)
    clf_struct = fum.KNN(k=2)
    clf_struct.fit(CATEGORICAL_STRUCT_ARRAY, CATEGORICAL_NP_ARRAY_TARGET)

    pd, linespace = ftmfi.partial_dependence(
        CATEGORICAL_NP_ARRAY_TEST, clf, 2, treat_as_categorical=True)
    assert np.allclose(pd, CATEGORICAL_NP_PD)
    assert np.array_equal(linespace, CATEGORICAL_NP_LINESPACE)

    pd, linespace = ftmfi.partial_dependence(
        CATEGORICAL_STRUCT_ARRAY_TEST,
        clf_struct,
        'c',
        treat_as_categorical=True)
    assert np.allclose(pd, CATEGORICAL_NP_PD)
    assert np.array_equal(linespace, CATEGORICAL_NP_LINESPACE)

    # Test row exclusion on a mixed array
    clf = fum.KNN(k=2)
    clf.fit(MIXED_ARRAY, MIXED_ARRAY_TARGET)

    pd, linespace = ftmfi.partial_dependence(
        MIXED_ARRAY_TEST,
        clf,
        'a',
        steps_number=3,
        include_rows=[0, 2],
        exclude_rows=1)
    assert np.allclose(pd, MIXED_PD_NUMERICAL)
    assert np.array_equal(linespace, MIXED_LINESPACE_NUMERICAL)

    pd, linespace = ftmfi.partial_dependence(
        MIXED_ARRAY_TEST, clf, 'b', exclude_rows=1)
    assert np.allclose(pd, MIXED_PD_CATEGORICAL)
    assert np.array_equal(linespace, MIXED_LINESPACE_CATEGORICAL)
