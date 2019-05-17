"""
Tests array tools.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import numpy as np

import pytest

import fatf.utils.array.tools as fuat
import fatf.utils.array.validation as fuav

from fatf.exceptions import IncorrectShapeError
from fatf.utils.testing.arrays import (
    BASE_NP_ARRAY, NOT_BASE_NP_ARRAY, NOT_BASE_STRUCTURED_ARRAY,
    NOT_NUMERICAL_NP_ARRAY, NOT_NUMERICAL_STRUCTURED_ARRAY, NUMERICAL_NP_ARRAY,
    NUMERICAL_STRUCTURED_ARRAY, WIDE_NP_ARRAY, WIDE_STRUCTURED_ARRAY)

NUMERICAL_UNSTRUCTURED_ARRAY = np.array([
    [1.0, 1.0 + 1j],
    [1, 1 + 1j],
    [np.nan, -1 + 1j],
    [np.inf, -1 + 1j],
    [-np.inf, -1 + 1j],
    [-1, -1 + 1j]])  # yapf: disable
NOT_NUMERICAL_UNSTRUCTURED_ARRAY = np.array([
    [1 + 0.j, 'a'],
    [1 + 0.j, 'b'],
    [-1 + 0.j, 'c'],
    [1 + 0.j, 'd'],
    [1 + 1j, 'e'],
    [0j, 'f'],
    [np.nan + 0j, 'g'],
    [np.inf + 0j, 'h'],
    [-np.inf + 0j, 'i']])  # yapf: disable
WIDE_UNSTRUCTURED_ARRAY = np.array([
    [1.0, 1.0 + 1j, np.nan],
    [np.inf, 1 + 1j, 6],
    [-1, -1 + 1j, -np.inf]])  # yapf: disable

NP_VER = [int(i) for i in np.version.version.split('.')]


def _compare_nan_arrays(array1, array2):
    """
    Compares 2 numpy arrays and returns True if they are element-wise the same.
    """
    assert not fuav.is_structured_array(array1), 'array1 cannot be structured.'
    assert not fuav.is_structured_array(array2), 'array2 cannot be structured.'
    assert array1.shape == array2.shape, 'Inputs must be of the same shape.'
    # pylint: disable=len-as-condition
    assert len(array1.shape) > 0 and len(array1.shape) < 3, 'Only 1D or 2D.'
    are_equal = True
    if len(array1.shape) == 1:
        for i in range(array1.shape[0]):
            if np.isnan(array1[i]) and np.isnan(array2[i]):
                continue
            elif array1[i] != array2[i]:
                are_equal = False
                break
    elif len(array1.shape) == 2:
        for i in range(array1.shape[0]):
            for j in range(array1.shape[1]):
                if np.isnan(array1[i, j]) and np.isnan(array2[i, j]):
                    continue
                elif array1[i, j] != array2[i, j]:
                    are_equal = False
                    break
            if not are_equal:
                break
    return are_equal


def test_compare_nan_arrays():
    """
    Tests numpy arrays element-wise array comparison.
    """
    assertion_error_1 = 'array1 cannot be structured.'
    assertion_error_2 = 'array2 cannot be structured.'
    assertion_error_3 = 'Inputs must be of the same shape.'
    assertion_error_4 = 'Only 1D or 2D.'

    array_3d_a = np.ones((2, 2, 2), dtype=float)
    array_struct = np.array([(1, 1.)], dtype=[('a', int), ('b', float)])
    array_1d_a = np.array([1, np.nan, 3])
    array_1d_b = np.array([-np.inf, 5, np.inf])
    array_2d_a = np.array([[1, np.nan, 3], [-np.inf, 5, np.inf]])
    array_2d_b = np.array([[-np.inf, 5, np.inf], [1, np.nan, 3]])

    # Assertion error 1 -- structured array 1
    with pytest.raises(AssertionError) as exin:
        _compare_nan_arrays(array_struct, array_struct)
    assert str(exin.value).startswith(assertion_error_1)

    # Assertion error 2 -- structured array 2
    with pytest.raises(AssertionError) as exin:
        _compare_nan_arrays(array_1d_a, array_struct)
    assert str(exin.value).startswith(assertion_error_2)

    # Assertion error 3 -- different shapes
    with pytest.raises(AssertionError) as exin:
        _compare_nan_arrays(array_1d_a, array_2d_a)
    assert str(exin.value).startswith(assertion_error_3)

    # Assertion error 4 -- 3D array
    with pytest.raises(AssertionError) as exin:
        _compare_nan_arrays(array_3d_a, array_3d_a)
    assert str(exin.value).startswith(assertion_error_4)

    # 1D
    assert not _compare_nan_arrays(array_1d_a, array_1d_b)
    assert _compare_nan_arrays(array_1d_a, array_2d_a[0, :])
    assert not _compare_nan_arrays(array_1d_a, array_2d_a[1, :])
    assert not _compare_nan_arrays(array_1d_b, array_2d_a[0, :])
    assert _compare_nan_arrays(array_1d_b, array_2d_a[1, :])

    # 2D
    assert not _compare_nan_arrays(array_2d_a, array_2d_b)
    assert not _compare_nan_arrays(array_2d_a[[0], :], array_2d_b[[0], :])
    assert _compare_nan_arrays(array_2d_a, array_2d_b[[1, 0], :])
    assert _compare_nan_arrays(array_2d_a[[0], :], array_2d_b[[1], :])
    assert _compare_nan_arrays(array_2d_a[[1], :], array_2d_b[[0], :])


def test_indices_by_type():
    """
    Tests :func:`fatf.utils.array.tools.indices_by_type` function.
    """
    # pylint: disable=too-many-locals,too-many-statements
    # Test any object and shape
    type_error = 'The input should be a numpy array-like.'
    incorrect_shape_error = 'The input array should be 2-dimensional.'
    value_error = ('indices_by_type only supports input arrays that hold base '
                   'numpy types, i.e. numerical and string-like -- numpy void '
                   'and object-like types are not allowed.')
    with pytest.raises(TypeError) as exin:
        fuat.indices_by_type(None)
    assert str(exin.value) == type_error
    with pytest.raises(IncorrectShapeError) as exin:
        fuat.indices_by_type(np.empty((0, )))
    assert str(exin.value) == incorrect_shape_error
    with pytest.raises(ValueError) as exin:
        fuat.indices_by_type(NOT_NUMERICAL_NP_ARRAY)
    assert str(exin.value) == value_error

    # Empty array
    i_n, i_c = fuat.indices_by_type(np.empty((22, 0)))
    assert np.array_equal([], i_n)
    assert np.array_equal([], i_c)

    # All numerical array
    array_all_numerical = np.ones((22, 4))
    array_all_numerical_indices_numerical = np.array([0, 1, 2, 3])
    array_all_numerical_indices_categorical = np.array([], dtype=int)
    i_n, i_c = fuat.indices_by_type(array_all_numerical)
    assert np.array_equal(array_all_numerical_indices_numerical, i_n)
    assert np.array_equal(array_all_numerical_indices_categorical, i_c)

    # All categorical -- single type -- array
    array_all_categorical = np.ones((22, 4), dtype='U4')
    array_all_categorical_indices_numerical = np.array([])
    array_all_categorical_indices_categorical = np.array([0, 1, 2, 3])
    i_n, i_c = fuat.indices_by_type(array_all_categorical)
    assert np.array_equal(array_all_categorical_indices_numerical, i_n)
    assert np.array_equal(array_all_categorical_indices_categorical, i_c)

    # Mixture array
    array_mixture_1 = np.ones((22, ), dtype=[('a', 'U4'),
                                             ('b', 'U4'),
                                             ('c', 'U4'),
                                             ('d', 'U4')])  # yapf: disable
    array_mixture_1_indices_numerical = np.array([])
    array_mixture_1_indices_categorical = np.array(['a', 'b', 'c', 'd'],
                                                   dtype='U1')
    ####
    i_n, i_c = fuat.indices_by_type(array_mixture_1)
    assert np.array_equal(array_mixture_1_indices_numerical, i_n)
    assert np.array_equal(array_mixture_1_indices_categorical, i_c)

    array_mixture_2 = np.ones((22, ), dtype=[('a', 'U4'),
                                             ('b', 'f'),
                                             ('c', 'U4'),
                                             ('d', int)])  # yapf: disable
    array_mixture_2_indices_numerical = np.array(['b', 'd'], dtype='U1')
    array_mixture_2_indices_categorical = np.array(['a', 'c'], dtype='U1')
    i_n, i_c = fuat.indices_by_type(array_mixture_2)
    assert np.array_equal(array_mixture_2_indices_numerical, i_n)
    assert np.array_equal(array_mixture_2_indices_categorical, i_c)

    glob_indices_numerical = np.array([0, 1])
    glob_indices_categorical = np.array([])
    i_n, i_c = fuat.indices_by_type(NUMERICAL_NP_ARRAY)
    assert np.array_equal(glob_indices_numerical, i_n)
    assert np.array_equal(glob_indices_categorical, i_c)
    #
    glob_indices_numerical = np.array([0, 1, 2])
    glob_indices_categorical = np.array([])
    i_n, i_c = fuat.indices_by_type(WIDE_NP_ARRAY)
    assert np.array_equal(glob_indices_numerical, i_n)
    assert np.array_equal(glob_indices_categorical, i_c)
    #
    glob_indices_numerical = np.array(['numbers', 'complex'])
    glob_indices_categorical = np.array([])
    i_n, i_c = fuat.indices_by_type(NUMERICAL_STRUCTURED_ARRAY)
    assert np.array_equal(glob_indices_numerical, i_n)
    assert np.array_equal(glob_indices_categorical, i_c)
    #
    glob_indices_numerical = np.array(['numerical'])
    glob_indices_categorical = np.array(['categorical'])
    i_n, i_c = fuat.indices_by_type(NOT_NUMERICAL_STRUCTURED_ARRAY)
    assert np.array_equal(glob_indices_numerical, i_n)
    assert np.array_equal(glob_indices_categorical, i_c)
    #
    glob_indices_numerical = np.array(['numbers', 'complex', 'anybody'])
    glob_indices_categorical = np.array([])
    i_n, i_c = fuat.indices_by_type(WIDE_STRUCTURED_ARRAY)
    assert np.array_equal(glob_indices_numerical, i_n)
    assert np.array_equal(glob_indices_categorical, i_c)


def test_get_invalid_indices():
    """
    Tests :func:`fatf.utils.array.tools.get_invalid_indices` function.
    """
    type_error = 'Input arrays should be numpy array-like objects.'
    incorrect_shape_array = 'The input array should be 2-dimensional.'
    incorrect_shape_indices = 'The indices array should be 1-dimensional.'
    with pytest.raises(TypeError) as exin:
        fuat.get_invalid_indices(None, np.ones((4, )))
    assert str(exin.value) == type_error
    with pytest.raises(TypeError) as exin:
        fuat.get_invalid_indices(None, np.ones((4, 4)))
    assert str(exin.value) == type_error
    with pytest.raises(TypeError) as exin:
        fuat.get_invalid_indices(np.ones((4, )), None)
    assert str(exin.value) == type_error
    with pytest.raises(TypeError) as exin:
        fuat.get_invalid_indices(None, np.ones((4, 4)))
    assert str(exin.value) == type_error
    # Incorrect shape array
    with pytest.raises(IncorrectShapeError) as exin:
        fuat.get_invalid_indices(np.ones((5, )), np.ones((4, 4)))
    assert str(exin.value) == incorrect_shape_array
    with pytest.raises(IncorrectShapeError) as exin:
        fuat.get_invalid_indices(np.ones((5, )), np.ones((4, )))
    assert str(exin.value) == incorrect_shape_array
    with pytest.raises(IncorrectShapeError) as exin:
        fuat.get_invalid_indices(np.ones((5, 3)), np.ones((4, 4)))
    assert str(exin.value) == incorrect_shape_indices

    gind = fuat.get_invalid_indices(NUMERICAL_NP_ARRAY, np.array([0, 2]))
    assert np.array_equal(gind, np.array([2]))
    gind = fuat.get_invalid_indices(NUMERICAL_NP_ARRAY, np.array(['a', 1]))
    assert np.array_equal(gind, np.array(['1', 'a']))
    gind = fuat.get_invalid_indices(NUMERICAL_NP_ARRAY, np.array([1, 0]))
    assert np.array_equal(gind, np.array([]))
    assert np.array_equal(gind, np.empty((0, )))
    #
    gind = fuat.get_invalid_indices(NOT_NUMERICAL_NP_ARRAY, np.array([0, 2]))
    assert np.array_equal(gind, np.array([2]))
    gind = fuat.get_invalid_indices(NOT_NUMERICAL_NP_ARRAY, np.array(['a', 1]))
    assert np.array_equal(gind, np.array(['1', 'a']))
    #
    gind = fuat.get_invalid_indices(NUMERICAL_STRUCTURED_ARRAY,
                                    np.array([0, 'numbers']))
    assert np.array_equal(gind, np.array(['0']))
    gind = fuat.get_invalid_indices(NUMERICAL_STRUCTURED_ARRAY, np.array([0]))
    assert np.array_equal(gind, np.array([0]))
    gind = fuat.get_invalid_indices(NUMERICAL_STRUCTURED_ARRAY,
                                    np.array(['complex', 'numbers']))
    assert np.array_equal(gind, np.array([]))
    #
    gind = fuat.get_invalid_indices(WIDE_STRUCTURED_ARRAY,
                                    np.array(['complex', 'numbers']))
    assert np.array_equal(gind, np.array([]))


def test_are_indices_valid():
    """
    Tests :func:`fatf.utils.array.tools.are_indices_valid` function.
    """
    type_error = 'Input arrays should be numpy array-like objects.'
    incorrect_shape_array = 'The input array should be 2-dimensional.'
    incorrect_shape_indices = 'The indices array should be 1-dimensional.'
    with pytest.raises(TypeError) as exin:
        fuat.are_indices_valid(None, np.ones((4, )))
    assert str(exin.value) == type_error
    with pytest.raises(TypeError) as exin:
        fuat.are_indices_valid(None, np.ones((4, 4)))
    assert str(exin.value) == type_error
    with pytest.raises(TypeError) as exin:
        fuat.are_indices_valid(np.ones((4, )), None)
    assert str(exin.value) == type_error
    with pytest.raises(TypeError) as exin:
        fuat.are_indices_valid(None, np.ones((4, 4)))
    assert str(exin.value) == type_error
    # Incorrect shape array
    with pytest.raises(IncorrectShapeError) as exin:
        fuat.are_indices_valid(np.ones((5, )), np.ones((4, 4)))
    assert str(exin.value) == incorrect_shape_array
    with pytest.raises(IncorrectShapeError) as exin:
        fuat.are_indices_valid(np.ones((5, )), np.ones((4, )))
    assert str(exin.value) == incorrect_shape_array
    with pytest.raises(IncorrectShapeError) as exin:
        fuat.are_indices_valid(np.ones((5, 3)), np.ones((4, 4)))
    assert str(exin.value) == incorrect_shape_indices

    assert not fuat.are_indices_valid(NUMERICAL_NP_ARRAY, np.array([0, 2]))
    assert not fuat.are_indices_valid(NUMERICAL_NP_ARRAY, np.array(['a', 1]))
    assert fuat.are_indices_valid(NUMERICAL_NP_ARRAY, np.array([1, 0]))
    #
    assert not fuat.are_indices_valid(NOT_NUMERICAL_NP_ARRAY, np.array([0, 2]))
    assert not fuat.are_indices_valid(NOT_NUMERICAL_NP_ARRAY,
                                      np.array(['a', 1]))  # yapf: disable
    assert fuat.are_indices_valid(NOT_NUMERICAL_NP_ARRAY, np.array([0, 1]))
    #
    assert not fuat.are_indices_valid(NUMERICAL_STRUCTURED_ARRAY,
                                      np.array([0, 'numbers']))
    assert not fuat.are_indices_valid(NUMERICAL_STRUCTURED_ARRAY,
                                      np.array([0]))  # yapf: disable
    assert fuat.are_indices_valid(NUMERICAL_STRUCTURED_ARRAY,
                                  np.array(['complex', 'numbers']))
    #
    assert fuat.are_indices_valid(WIDE_STRUCTURED_ARRAY,
                                  np.array(['complex', 'numbers']))


def test_generalise_dtype():
    """
    Tests :func:`fatf.utils.array.tools.generalise_dtype`.
    """
    error_msg = 'The {} dtype is not one of the base types (strings/numbers).'
    with pytest.raises(ValueError) as exin:
        fuat.generalise_dtype(np.dtype(np.datetime64), np.dtype(np.datetime64))
    assert str(exin.value) == error_msg.format('first')

    with pytest.raises(ValueError) as exin:
        fuat.generalise_dtype(np.dtype(np.float64), np.dtype(np.datetime64))
    assert str(exin.value) == error_msg.format('second')

    dtype_int = np.dtype(int)
    dtype_int32 = np.dtype(np.int32)
    dtype_int64 = np.dtype(np.int64)
    dtype_float = np.dtype(float)
    dtype_float16 = np.dtype(np.float16)
    dtype_float32 = np.dtype(np.float32)
    dtype_float64 = np.dtype(np.float64)
    dtype_str = np.dtype(str)
    dtype_str4 = np.dtype('U4')
    dtype_str11 = np.dtype('U11')
    dtype_str16 = np.dtype('U16')
    dtype_str21 = np.dtype('U21')
    dtype_str32 = np.dtype('U32')

    assert dtype_int64 is fuat.generalise_dtype(dtype_int, dtype_int32)
    assert dtype_int64 is fuat.generalise_dtype(dtype_int, dtype_int64)
    assert dtype_int64 is fuat.generalise_dtype(dtype_int32, dtype_int64)
    assert dtype_int64 is fuat.generalise_dtype(dtype_int, dtype_int)

    assert dtype_float64 is fuat.generalise_dtype(dtype_float, dtype_float)
    assert dtype_float64 is fuat.generalise_dtype(dtype_float64, dtype_float)
    assert dtype_float64 is fuat.generalise_dtype(dtype_int, dtype_float32)
    assert dtype_float64 is fuat.generalise_dtype(dtype_int32, dtype_float32)
    assert dtype_float32 is fuat.generalise_dtype(dtype_float32, dtype_float16)

    assert dtype_str4 is fuat.generalise_dtype(dtype_str, dtype_str4)
    assert dtype_str21 is fuat.generalise_dtype(dtype_str21, dtype_str4)

    assert dtype_str16 == fuat.generalise_dtype(dtype_str11, dtype_str16)
    assert dtype_str11 == fuat.generalise_dtype(dtype_int32, dtype_str4)
    assert dtype_str21 == fuat.generalise_dtype(dtype_int64, dtype_str4)
    assert dtype_str32 == fuat.generalise_dtype(dtype_float32, dtype_str4)
    assert dtype_str32 == fuat.generalise_dtype(dtype_float64, dtype_str16)


def test_fatf_structured_to_unstructured_row():
    """
    Tests :func:`fatf.utils.array.tools.fatf_structured_to_unstructured_row`.
    """
    type_error = 'The input should be a row of a structured array.'
    with pytest.raises(TypeError) as exin:
        fuat.fatf_structured_to_unstructured_row(
            np.array([b'123'], np.void)[0])
    assert str(exin.value) == type_error
    value_arror = ('structured_to_unstructured_row only supports conversion '
                   'of structured rows that hold base numpy types, i.e. '
                   'numerical and string-like -- numpy void and object-like '
                   'types are not allowed.')
    with pytest.raises(ValueError) as exin:
        fuat.fatf_structured_to_unstructured_row(NOT_BASE_STRUCTURED_ARRAY[0])
    assert str(exin.value) == value_arror

    simple = fuat.fatf_structured_to_unstructured_row(
        NUMERICAL_STRUCTURED_ARRAY[0])
    assert _compare_nan_arrays(simple, NUMERICAL_UNSTRUCTURED_ARRAY[0])
    simple = fuat.fatf_structured_to_unstructured_row(
        NUMERICAL_STRUCTURED_ARRAY[2])
    assert _compare_nan_arrays(simple, NUMERICAL_UNSTRUCTURED_ARRAY[2])
    simple = fuat.fatf_structured_to_unstructured_row(
        NUMERICAL_STRUCTURED_ARRAY[3])
    assert _compare_nan_arrays(simple, NUMERICAL_UNSTRUCTURED_ARRAY[3])
    #
    simple = fuat.fatf_structured_to_unstructured_row(
        NOT_NUMERICAL_STRUCTURED_ARRAY[0])
    assert np.array_equal(simple, NOT_NUMERICAL_UNSTRUCTURED_ARRAY[0])
    simple = fuat.fatf_structured_to_unstructured_row(
        NOT_NUMERICAL_STRUCTURED_ARRAY[6])
    assert np.array_equal(simple, NOT_NUMERICAL_UNSTRUCTURED_ARRAY[6])
    simple = fuat.fatf_structured_to_unstructured_row(
        NOT_NUMERICAL_STRUCTURED_ARRAY[7])
    assert np.array_equal(simple, NOT_NUMERICAL_UNSTRUCTURED_ARRAY[7])
    #
    simple = fuat.fatf_structured_to_unstructured_row(WIDE_STRUCTURED_ARRAY[0])
    assert _compare_nan_arrays(simple, WIDE_UNSTRUCTURED_ARRAY[0])
    simple = fuat.fatf_structured_to_unstructured_row(WIDE_STRUCTURED_ARRAY[2])
    assert _compare_nan_arrays(simple, WIDE_UNSTRUCTURED_ARRAY[2])

    assert fuat.fatf_structured_to_unstructured_row(
        np.array([(7, )], dtype=[('f', float)])[0]) == 7


def test_structured_to_unstructured_row():
    """
    Tests :func:`fatf.utils.array.tools.structured_to_unstructured_row`.
    """
    simple = fuat.fatf_structured_to_unstructured_row(
        NUMERICAL_STRUCTURED_ARRAY[2])
    assert _compare_nan_arrays(simple, NUMERICAL_UNSTRUCTURED_ARRAY[2])
    assert fuat.fatf_structured_to_unstructured_row(
        np.array([(7, )], dtype=[('f', float)])[0]) == 7
    assert ('This function need not be tested as test_choose_structured_to_'
            'unstructured and test_fatf_structured_to_unstructured_row tests '
            'are sufficient and there is no straight forward way of testing '
            'it.')


def test_choose_structured_to_unstructured(caplog):
    """
    Tests :func:`fatf.utils.array.tools._choose_structured_to_unstructured`.
    """
    # pylint: disable=protected-access
    # Memorise current numpy version
    installed_numpy_version = np.version.version
    # Fake version lower than 1.16.0
    np.version.version = '1.15.999'
    log_message = ("Using fatf's fatf.utils.array.tools."
                   'fatf_structured_to_unstructured as fatf.utils.'
                   'array.tools.structured_to_unstructured and fatf.utils.'
                   'array.tools.fatf_structured_to_unstructured_row as '
                   'fatf.utils.array.tools.structured_to_unstructured_row.')
    assert fuat._choose_structured_to_unstructured()
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == 'INFO'
    assert caplog.records[0].getMessage() == log_message
    # Fake at least 1.16.0 version
    np.version.version = '1.16.000'
    log_message = ("Using numpy's numpy.lib.recfunctions."
                   'structured_to_unstructured as fatf.utils.array.tools.'
                   'structured_to_unstructured and fatf.utils.array.tools.'
                   'structured_to_unstructured_row.')
    assert not fuat._choose_structured_to_unstructured()
    assert len(caplog.records) == 2
    assert caplog.records[1].levelname == 'INFO'
    assert caplog.records[1].getMessage() == log_message
    # Restore numpy version
    assert len(caplog.records) == 2
    np.version.version = installed_numpy_version


def test_fatf_structured_to_unstructured():
    """
    Tests :func:`fatf.utils.array.tools.fatf_structured_to_unstructured`.
    """
    # Wrong array types
    type_error = 'structured_array should be a structured numpy array.'
    with pytest.raises(TypeError) as exin:
        fuat.fatf_structured_to_unstructured(NUMERICAL_NP_ARRAY)
    assert str(exin.value) == type_error
    with pytest.raises(TypeError) as exin:
        fuat.fatf_structured_to_unstructured(NOT_NUMERICAL_NP_ARRAY)
    assert str(exin.value) == type_error
    with pytest.raises(TypeError) as exin:
        fuat.fatf_structured_to_unstructured(WIDE_NP_ARRAY)
    assert str(exin.value) == type_error

    # Arrays of complex-types
    value_error = ('fatf_structured_to_unstructured only supports conversion '
                   'of arrays that hold base numpy types, i.e. numerical and '
                   'string-like -- numpy void and object-like types are not '
                   'allowed.')
    complex_array = np.array([(None, object())],
                             dtype=[('n', 'O'), ('o', 'O')])
    with pytest.raises(ValueError) as exin:
        fuat.fatf_structured_to_unstructured(complex_array)
    assert str(exin.value) == value_error

    # Right type
    simple = fuat.fatf_structured_to_unstructured(NUMERICAL_STRUCTURED_ARRAY)
    assert _compare_nan_arrays(simple, NUMERICAL_UNSTRUCTURED_ARRAY)
    simple = fuat.fatf_structured_to_unstructured(
        NOT_NUMERICAL_STRUCTURED_ARRAY)
    assert np.array_equal(simple, NOT_NUMERICAL_UNSTRUCTURED_ARRAY)
    simple = fuat.fatf_structured_to_unstructured(WIDE_STRUCTURED_ARRAY)
    assert _compare_nan_arrays(simple, WIDE_UNSTRUCTURED_ARRAY)

    simple = fuat.fatf_structured_to_unstructured(
        np.array([(7, )], dtype=[('f', float)]))
    assert np.array_equal(simple, np.array([[7]]))
    simple = fuat.fatf_structured_to_unstructured(
        np.array([(4, ), (2, )], dtype=[('f', float)]))
    assert np.array_equal(simple, np.array([[4], [2]]))


def test_structured_to_unstructured():
    """
    Tests :func:`fatf.utils.array.tools.structured_to_unstructured`.
    """
    simple = fuat.structured_to_unstructured(NOT_NUMERICAL_STRUCTURED_ARRAY)
    assert np.array_equal(simple, NOT_NUMERICAL_UNSTRUCTURED_ARRAY)
    simple = fuat.structured_to_unstructured(
        np.array([(7, )], dtype=[('f', float)]))
    assert _compare_nan_arrays(simple, np.array([[7]]))
    assert ('This function need not be tested as test_choose_structured_to_'
            'unstructured and test_fatf_structured_to_unstructured tests are '
            'sufficient and there is no straight forward way of testing it.')


def test_as_unstructured():
    """
    Tests :func:`fatf.utils.array.tools.as_unstructured`.
    """
    type_error = ('The input should either be a numpy (structured or '
                  'unstructured) array-like object (numpy.ndarray) or a row '
                  'of a structured numpy array (numpy.void).')
    value_error = ('as_unstructured only supports conversion of arrays that '
                   'hold base numpy types, i.e. numerical and string-like -- '
                   'numpy void and object-like types are not allowed.')
    # Test incompatible -- None -- type
    with pytest.raises(TypeError) as exin:
        fuat.as_unstructured(None)
    assert str(exin.value) == type_error

    # Test np.void -- a structured array's row
    simple = fuat.as_unstructured(NUMERICAL_STRUCTURED_ARRAY[0])
    assert _compare_nan_arrays(simple, NUMERICAL_UNSTRUCTURED_ARRAY[0])

    # Test structured array
    simple = fuat.as_unstructured(NOT_NUMERICAL_STRUCTURED_ARRAY)
    assert np.array_equal(simple, NOT_NUMERICAL_UNSTRUCTURED_ARRAY)
    # Test unstructured -- base type
    simple = fuat.as_unstructured(BASE_NP_ARRAY)
    assert np.array_equal(simple, BASE_NP_ARRAY)
    # Test unstructured -- not base type
    with pytest.raises(ValueError) as exin:
        fuat.as_unstructured(NOT_BASE_NP_ARRAY)
    assert str(exin.value) == value_error
