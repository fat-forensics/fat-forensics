"""
Tests functions responsible for objects validation across FAT Forensics.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

# pylint: disable=too-many-lines

import numpy as np
import pytest

import fatf.utils.array.validation as fuav
import fatf.utils.tools as fut

from fatf.utils.testing.arrays import (
    NUMERICAL_NP_ARRAY, NOT_NUMERICAL_NP_ARRAY, WIDE_NP_ARRAY,
    NUMERICAL_STRUCTURED_ARRAY, NOT_NUMERICAL_STRUCTURED_ARRAY,
    WIDE_STRUCTURED_ARRAY, BASE_NP_ARRAY, NOT_BASE_NP_ARRAY,
    BASE_STRUCTURED_ARRAY, NOT_BASE_STRUCTURED_ARRAY)

NUMERICAL_KINDS = [True, 1, -1, 1.0, 1 + 1j, np.nan, np.inf, -np.inf]
NOT_NUMERICAL_KINDS = [object(), 'string', u'unicode', None]
TEXTUAL_KINDS = ['string', u'unicode']
UNSUPPORTED_TEXTUAL_KINDS = [b'bytes']
UNSUPPORTED_TEXTUAL_DTYPES = [np.dtype('S'), np.dtype('a')]
BASE_KINDS = [True, 1, -1, 1.0, 1 + 1j, 'string', u'unicode', b'bytes', np.nan,
              np.inf, -np.inf]  # yapf: disable
NOT_BASE_KINDS = [None, object()]

NP_VER = [int(i) for i in np.version.version.split('.')]
NP_VER_TYPEERROR_MSG_14 = 'a bytes-like object is required, not \'int\''
NP_VER_TYPEERROR_MSG_12 = 'Empty data-type'


def test_is_numerical_dtype():
    """
    Tests :func:`fatf.utils.array.validation.is_numerical_dtype` function.
    """
    type_error = 'The input should be a numpy dtype object.'
    value_error = ('The numpy dtype object is structured. '
                   'Only base dtype are allowed.')
    # Test any object
    with pytest.raises(TypeError) as exin:
        fuav.is_numerical_dtype(None)
    assert str(exin.value) == type_error

    # Test simple numerical arrays
    for i in NUMERICAL_KINDS:
        for j in [i, [i], [i] * 2, [[i] * 2] * 2]:
            array = np.array(j)
            array_dtype = array.dtype
            assert fuav.is_numerical_dtype(array_dtype) is True

    # Test simple not numerical arrays
    for i in NOT_NUMERICAL_KINDS:
        for j in [i, [i], [i] * 2, [[i] * 2] * 2]:
            array = np.array(j)
            array_dtype = array.dtype
            assert fuav.is_numerical_dtype(array_dtype) is False

    # Test simple numerical array
    assert fuav.is_numerical_dtype(NUMERICAL_NP_ARRAY.dtype) is True
    # Test simple not numerical array
    assert fuav.is_numerical_dtype(NOT_NUMERICAL_NP_ARRAY.dtype) is False
    assert fuav.is_numerical_dtype(BASE_NP_ARRAY.dtype) is False
    assert fuav.is_numerical_dtype(NOT_BASE_NP_ARRAY.dtype) is False

    # Test structured numerical array
    with pytest.raises(ValueError) as exin:
        fuav.is_numerical_dtype(NUMERICAL_STRUCTURED_ARRAY.dtype)
    assert str(exin.value) == value_error
    # Test structured not numerical array
    with pytest.raises(ValueError) as exin:
        fuav.is_numerical_dtype(NOT_NUMERICAL_STRUCTURED_ARRAY.dtype)
    assert str(exin.value) == value_error
    with pytest.raises(ValueError) as exin:
        fuav.is_numerical_dtype(BASE_STRUCTURED_ARRAY.dtype)
    assert str(exin.value) == value_error
    with pytest.raises(ValueError) as exin:
        fuav.is_numerical_dtype(NOT_BASE_STRUCTURED_ARRAY.dtype)
    assert str(exin.value) == value_error

    # Test numpy types
    for kind, dtypes in np.sctypes.items():
        if kind == 'others':
            for dtype in dtypes:
                if dtype is bool:
                    assert fuav.is_numerical_dtype(np.dtype(dtype)) is True
                else:
                    assert fuav.is_numerical_dtype(np.dtype(dtype)) is False
        else:
            for dtype in dtypes:
                assert fuav.is_numerical_dtype(np.dtype(dtype)) is True


def test_is_textual_dtype():
    """
    Tests :func:`fatf.utils.array.validation.is_textual_dtype` function.
    """
    # pylint: disable=too-many-branches,too-many-statements
    type_error = 'The input should be a numpy dtype object.'
    value_error = ('The numpy dtype object is structured. '
                   'Only base dtype are allowed.')
    warning_message = ('Zero-terminated bytes type is not supported and is '
                       'not considered to be a textual type. Please use any '
                       'other textual type.')
    # Test any object
    with pytest.raises(TypeError) as exin:
        fuav.is_textual_dtype(None)
    assert str(exin.value) == type_error

    # Test simple numerical arrays
    for i in NUMERICAL_KINDS:
        for j in [i, [i], [i] * 2, [[i] * 2] * 2]:
            array = np.array(j)
            array_dtype = array.dtype
            assert fuav.is_textual_dtype(array_dtype) is False

    # Test simple textual arrays
    for i in TEXTUAL_KINDS:
        for j in [i, [i], [i] * 2, [[i] * 2] * 2]:
            array = np.array(j)
            array_dtype = array.dtype
            assert fuav.is_textual_dtype(array_dtype) is True

    # Test simple not numerical arrays
    for i in NOT_BASE_KINDS:
        for j in [i, [i], [i] * 2, [[i] * 2] * 2]:
            array = np.array(j)
            array_dtype = array.dtype
            assert fuav.is_textual_dtype(array_dtype) is False

    for i in UNSUPPORTED_TEXTUAL_KINDS:
        for j in [i, [i], [i] * 2, [[i] * 2] * 2]:
            array = np.array(j)
            array_dtype = array.dtype
            with pytest.warns(UserWarning) as warning:
                assert fuav.is_textual_dtype(array_dtype) is False
            assert warning_message == str(warning[0].message)

    for dtype in UNSUPPORTED_TEXTUAL_DTYPES:
        with pytest.warns(UserWarning) as warning:
            assert fuav.is_textual_dtype(dtype) is False
        assert warning_message == str(warning[0].message)

    # Test simple numerical array
    assert fuav.is_textual_dtype(NUMERICAL_NP_ARRAY.dtype) is False
    # Test simple not numerical array (with objects)
    assert fuav.is_textual_dtype(NOT_NUMERICAL_NP_ARRAY.dtype) is False
    assert fuav.is_textual_dtype(BASE_NP_ARRAY.dtype) is True
    assert fuav.is_textual_dtype(NOT_BASE_NP_ARRAY.dtype) is False

    # Test structured numerical array
    with pytest.raises(ValueError) as exin:
        fuav.is_textual_dtype(NUMERICAL_STRUCTURED_ARRAY.dtype)
    assert str(exin.value) == value_error
    # Test structured not numerical array
    with pytest.raises(ValueError) as exin:
        fuav.is_textual_dtype(NOT_NUMERICAL_STRUCTURED_ARRAY.dtype)
    assert str(exin.value) == value_error
    with pytest.raises(ValueError) as exin:
        fuav.is_textual_dtype(BASE_STRUCTURED_ARRAY.dtype)
    assert str(exin.value) == value_error
    with pytest.raises(ValueError) as exin:
        fuav.is_textual_dtype(NOT_BASE_STRUCTURED_ARRAY.dtype)
    assert str(exin.value) == value_error

    # Test numpy types
    for kind, dtypes in np.sctypes.items():
        if kind == 'others':
            for dtype in dtypes:
                if dtype is str:
                    assert fuav.is_textual_dtype(np.dtype(dtype)) is True
                elif dtype is bytes:  # pragma: no cover
                    with pytest.warns(UserWarning) as warning:
                        assert fuav.is_textual_dtype(np.dtype(dtype)) is False
                    assert warning_message == str(warning[0].message)
                else:
                    assert fuav.is_textual_dtype(np.dtype(dtype)) is False
        else:
            for dtype in dtypes:
                assert fuav.is_textual_dtype(np.dtype(dtype)) is False


def test_is_base_dtype():
    """
    Tests :func:`fatf.utils.array.validation.is_base_dtype` function.
    """
    type_error = 'The input should be a numpy dtype object.'
    value_error = ('The numpy dtype object is structured. '
                   'Only base dtype are allowed.')
    # Test any object
    with pytest.raises(TypeError) as exin:
        fuav.is_base_dtype(None)
    assert str(exin.value) == type_error

    # Test simple type arrays
    for i in BASE_KINDS:
        for j in [i, [i], [i] * 2, [[i] * 2] * 2]:
            array = np.array(j)
            array_dtype = array.dtype
            assert fuav.is_base_dtype(array_dtype) is True

    # Test simple not numerical arrays
    for i in NOT_BASE_KINDS:
        for j in [i, [i], [i] * 2, [[i] * 2] * 2]:
            array = np.array(j)
            array_dtype = array.dtype
            assert fuav.is_base_dtype(array_dtype) is False

    # Test simple array
    assert fuav.is_base_dtype(NUMERICAL_NP_ARRAY.dtype) is True
    assert fuav.is_base_dtype(NOT_NUMERICAL_NP_ARRAY.dtype) is False
    assert fuav.is_base_dtype(BASE_NP_ARRAY.dtype) is True
    assert fuav.is_base_dtype(NOT_BASE_NP_ARRAY.dtype) is False

    # Test structured array
    with pytest.raises(ValueError) as exin:
        fuav.is_base_dtype(NUMERICAL_STRUCTURED_ARRAY.dtype)
    assert str(exin.value) == value_error
    with pytest.raises(ValueError) as exin:
        fuav.is_base_dtype(NOT_NUMERICAL_STRUCTURED_ARRAY.dtype)
    assert str(exin.value) == value_error
    with pytest.raises(ValueError) as exin:
        fuav.is_base_dtype(BASE_STRUCTURED_ARRAY.dtype)
    assert str(exin.value) == value_error
    with pytest.raises(ValueError) as exin:
        fuav.is_base_dtype(NOT_BASE_STRUCTURED_ARRAY.dtype)
    assert str(exin.value) == value_error

    # Test numpy types
    for kind, dtypes in np.sctypes.items():
        if kind == 'others':
            for dtype in dtypes:
                if dtype is bool or dtype is str or dtype is bytes:
                    assert fuav.is_base_dtype(np.dtype(dtype)) is True
                else:
                    assert fuav.is_base_dtype(np.dtype(dtype)) is False
        else:
            for dtype in dtypes:
                assert fuav.is_base_dtype(np.dtype(dtype)) is True


def test_is_flat_dtype():
    """
    Tests :func:`fatf.utils.array.validation.is_flat_dtype` function.
    """

    def numpy_low():
        assert fuav.is_flat_dtype(NUMERICAL_NP_ARRAY.dtype)
        assert fuav.is_flat_dtype(NUMERICAL_STRUCTURED_ARRAY.dtype[0])
        assert fuav.is_flat_dtype(weird_array_1.dtype[0])
        assert fuav.is_flat_dtype(weird_array_1.dtype[1])
        assert not fuav.is_flat_dtype(weird_array_1.dtype[2])
        assert fuav.is_flat_dtype(weird_array_2.dtype)

    def numpy_high():  # pragma: no cover
        assert fuav.is_flat_dtype(NUMERICAL_NP_ARRAY.dtype)
        assert fuav.is_flat_dtype(NUMERICAL_STRUCTURED_ARRAY.dtype[0])
        assert fuav.is_flat_dtype(weird_array_1.dtype[0])
        assert fuav.is_flat_dtype(weird_array_1.dtype[1])
        assert not fuav.is_flat_dtype(weird_array_1.dtype[2])
        assert fuav.is_flat_dtype(weird_array_2.dtype)

    type_error = 'The input should be a numpy dtype object.'
    value_error = ('The numpy dtype object is structured. '
                   'Only base dtype are allowed.')
    # Test any object
    with pytest.raises(TypeError) as exin:
        fuav.is_flat_dtype(None)
    assert str(exin.value) == type_error

    # Test structured array
    with pytest.raises(ValueError) as exin:
        fuav.is_flat_dtype(NUMERICAL_STRUCTURED_ARRAY.dtype)
    assert str(exin.value) == value_error
    with pytest.raises(ValueError) as exin:
        fuav.is_flat_dtype(NOT_NUMERICAL_STRUCTURED_ARRAY.dtype)
    assert str(exin.value) == value_error
    with pytest.raises(ValueError) as exin:
        fuav.is_flat_dtype(BASE_STRUCTURED_ARRAY.dtype)
    assert str(exin.value) == value_error
    with pytest.raises(ValueError) as exin:
        fuav.is_flat_dtype(NOT_BASE_STRUCTURED_ARRAY.dtype)
    assert str(exin.value) == value_error

    weird_array_1 = np.zeros(
        3, dtype=[('x', 'f4'), ('y', np.float32), ('v', 'f4', (2, 2))])
    weird_array_2 = np.ones((2, 2), dtype=weird_array_1.dtype[2])

    if fuav._NUMPY_1_13:  # pragma: no cover # pylint: disable=protected-access
        numpy_low()
        numpy_high()
    else:  # pragma: no cover
        numpy_low()


def test_are_similar_dtypes():
    """
    Tests :func:`fatf.utils.array.validation.are_similar_dtypes` function.
    """
    # pylint: disable=too-many-statements
    type_error_a = 'dtype_a should be a numpy dtype object.'
    type_error_b = 'dtype_b should be a numpy dtype object.'
    value_error_a = ('The dtype_a is a structured numpy dtype object. Only '
                     'base dtype are allowed.')
    value_error_b = ('The dtype_b is a structured numpy dtype object. Only '
                     'base dtype are allowed.')

    simple_dtype = NUMERICAL_NP_ARRAY.dtype
    structured_dtype = NUMERICAL_STRUCTURED_ARRAY.dtype

    # Test any object
    with pytest.raises(TypeError) as exin:
        fuav.are_similar_dtypes(None, None, False)
    assert str(exin.value) == type_error_a
    with pytest.raises(TypeError) as exin:
        fuav.are_similar_dtypes(None, simple_dtype, True)
    assert str(exin.value) == type_error_a
    with pytest.raises(TypeError) as exin:
        fuav.are_similar_dtypes(simple_dtype, None, False)
    assert str(exin.value) == type_error_b
    with pytest.raises(TypeError) as exin:
        fuav.are_similar_dtypes(structured_dtype, None, True)
    assert str(exin.value) == type_error_b

    # Test structured dtype
    with pytest.raises(ValueError) as exin:
        fuav.are_similar_dtypes(structured_dtype, structured_dtype, True)
    assert str(exin.value) == value_error_a
    with pytest.raises(ValueError) as exin:
        fuav.are_similar_dtypes(structured_dtype, simple_dtype, False)
    assert str(exin.value) == value_error_a
    with pytest.raises(ValueError) as exin:
        fuav.are_similar_dtypes(simple_dtype, structured_dtype, True)
    assert str(exin.value) == value_error_b

    f1_dtype = np.array([5, 1.222]).dtype
    f2_dtype = np.array([5, 1], dtype=float).dtype
    f3_dtype = np.array([5, 1]).dtype
    c1_dtype = np.array(['a', 'b']).dtype
    c2_dtype = np.array(['a']).dtype
    c3_dtype = np.array(['ab']).dtype
    c4_dtype = np.array(['a'], dtype=str).dtype
    c5_dtype = np.array([u'a']).dtype

    # Strict type comparison
    assert fuav.are_similar_dtypes(f1_dtype, f2_dtype, True) is True
    assert fuav.are_similar_dtypes(f2_dtype, f3_dtype, True) is False
    assert fuav.are_similar_dtypes(f3_dtype, c1_dtype, True) is False
    assert fuav.are_similar_dtypes(c1_dtype, c2_dtype, True) is True
    assert fuav.are_similar_dtypes(c2_dtype, c4_dtype, True) is True
    assert fuav.are_similar_dtypes(c2_dtype, c3_dtype, True) is False
    assert fuav.are_similar_dtypes(c3_dtype, c4_dtype, True) is False
    assert fuav.are_similar_dtypes(c1_dtype, c5_dtype, True) is True
    assert fuav.are_similar_dtypes(c2_dtype, c5_dtype, True) is True

    # Fuzzy type comparison
    assert fuav.are_similar_dtypes(f1_dtype, f2_dtype, False) is True
    assert fuav.are_similar_dtypes(f2_dtype, f3_dtype, False) is True
    assert fuav.are_similar_dtypes(f3_dtype, c1_dtype, False) is False
    assert fuav.are_similar_dtypes(c1_dtype, c2_dtype, False) is True
    assert fuav.are_similar_dtypes(c2_dtype, c4_dtype, False) is True
    assert fuav.are_similar_dtypes(c2_dtype, c3_dtype, False) is True
    assert fuav.are_similar_dtypes(c3_dtype, c4_dtype, False) is True
    assert fuav.are_similar_dtypes(c1_dtype, c5_dtype, False) is True
    assert fuav.are_similar_dtypes(c2_dtype, c5_dtype, False) is True


def test_are_similar_dtype_arrays():
    """
    Tests :func:`fatf.utils.array.validation.are_similar_dtype_arrays`.
    """
    type_error_a = 'array_a should be a numpy array-like object.'
    type_error_b = 'array_b should be a numpy array-like object.'

    # Test any object
    with pytest.raises(TypeError) as exin:
        fuav.are_similar_dtype_arrays(None, None, False)
    assert str(exin.value) == type_error_a
    with pytest.raises(TypeError) as exin:
        fuav.are_similar_dtype_arrays(None, NUMERICAL_NP_ARRAY, True)
    assert str(exin.value) == type_error_a
    with pytest.raises(TypeError) as exin:
        fuav.are_similar_dtype_arrays(NUMERICAL_NP_ARRAY, None, False)
    assert str(exin.value) == type_error_b

    # One structured the other one unstructured
    assert fuav.are_similar_dtype_arrays(
        NUMERICAL_NP_ARRAY, NUMERICAL_STRUCTURED_ARRAY, False) is False
    assert fuav.are_similar_dtype_arrays(NUMERICAL_STRUCTURED_ARRAY,
                                         NUMERICAL_NP_ARRAY, True) is False

    f1_array = np.array([5, 1.222])
    f2_array = np.array([5, 1], dtype=float)
    f3_array = np.array([5, 1])
    c1_array = np.array(['a', 'b'])
    c2_array = np.array(['a'])
    c3_array = np.array(['ab'])
    c4_array = np.array(['a'], dtype=str)

    # Both unstructured
    # Strict type comparison
    assert fuav.are_similar_dtype_arrays(f1_array, f2_array, True) is True
    assert fuav.are_similar_dtype_arrays(f2_array, f3_array, True) is False
    assert fuav.are_similar_dtype_arrays(f3_array, c1_array, True) is False
    assert fuav.are_similar_dtype_arrays(c1_array, c2_array, True) is True
    assert fuav.are_similar_dtype_arrays(c2_array, c4_array, True) is True
    assert fuav.are_similar_dtype_arrays(c2_array, c3_array, True) is False
    assert fuav.are_similar_dtype_arrays(c3_array, c4_array, True) is False
    # Fuzzy type comparison
    assert fuav.are_similar_dtype_arrays(f1_array, f2_array, False) is True
    assert fuav.are_similar_dtype_arrays(f2_array, f3_array, False) is True
    assert fuav.are_similar_dtype_arrays(f3_array, c1_array, False) is False
    assert fuav.are_similar_dtype_arrays(c1_array, c2_array, False) is True
    assert fuav.are_similar_dtype_arrays(c2_array, c4_array, False) is True
    assert fuav.are_similar_dtype_arrays(c2_array, c3_array, False) is True
    assert fuav.are_similar_dtype_arrays(c3_array, c4_array, False) is True

    s1_array = np.array([(1, 'abc', 3.14)],
                        dtype=[('a', int), ('b', str), ('c', float)])
    s2_array = np.array([(1, 'abc')], dtype=[('a', int), ('b', str)])
    s3_array = np.array([(1, 'abc')], dtype=[('a', int), ('c', str)])
    s4_array = np.array([(1, 'abc')], dtype=[('a', int), ('b', str)])
    s5_array = np.array([(1, 'abc')], dtype=[('a', float), ('c', str)])

    # Both structured
    # Strict type comparison
    assert fuav.are_similar_dtype_arrays(s1_array, s2_array, True) is False
    assert fuav.are_similar_dtype_arrays(s2_array, s3_array, True) is False
    assert fuav.are_similar_dtype_arrays(s2_array, s4_array, True) is True
    assert fuav.are_similar_dtype_arrays(s4_array, s5_array, True) is False
    # Fuzzy type comparison
    assert fuav.are_similar_dtype_arrays(s1_array, s3_array, False) is False
    assert fuav.are_similar_dtype_arrays(s2_array, s3_array, False) is False
    assert fuav.are_similar_dtype_arrays(s2_array, s4_array, False) is True
    assert fuav.are_similar_dtype_arrays(s4_array, s5_array, False) is False


def test_is_numerical_array():
    """
    Tests :func:`fatf.utils.array.validation.is_numerical_array` function.
    """
    # pylint: disable=too-many-branches,too-many-statements
    type_error = 'The input should be a numpy array-like object.'
    # Test any object
    with pytest.raises(TypeError) as exin:
        fuav.is_numerical_array(None)
    assert str(exin.value) == type_error

    # Test simple numerical arrays
    for i in NUMERICAL_KINDS:
        for j in [i, [i], [i] * 2, [[i] * 2] * 2]:
            array = np.array(j)
            assert fuav.is_numerical_array(array) is True

    # Test simple not numerical arrays
    for i in NOT_NUMERICAL_KINDS:
        for j in [i, [i], [i] * 2, [[i] * 2] * 2]:
            array = np.array(j)
            assert fuav.is_numerical_array(array) is False

    # Test simple numerical array
    assert fuav.is_numerical_array(NUMERICAL_NP_ARRAY) is True
    # Test structured numerical array
    assert fuav.is_numerical_array(NUMERICAL_STRUCTURED_ARRAY) is True
    #
    assert fuav.is_numerical_array(WIDE_NP_ARRAY) is True
    assert fuav.is_numerical_array(WIDE_STRUCTURED_ARRAY) is True

    # Test simple not numerical array
    assert fuav.is_numerical_array(NOT_NUMERICAL_NP_ARRAY) is False
    # Test structured not numerical array
    assert fuav.is_numerical_array(NOT_NUMERICAL_STRUCTURED_ARRAY) is False

    # Test base arrays
    assert fuav.is_numerical_array(BASE_NP_ARRAY) is False
    assert fuav.is_numerical_array(NOT_BASE_NP_ARRAY) is False
    assert fuav.is_numerical_array(BASE_STRUCTURED_ARRAY) is False
    assert fuav.is_numerical_array(NOT_BASE_STRUCTURED_ARRAY) is False

    # Test numpy types
    for kind, dtypes in np.sctypes.items():
        # yapf: disable
        if kind == 'others':
            for dtype in dtypes:
                if dtype is bool:
                    assert fuav.is_numerical_array(
                        np.empty((1, ), dtype=dtype)) is True
                    assert fuav.is_numerical_array(
                        np.ones((1, ), dtype=dtype)) is True
                    assert fuav.is_numerical_array(
                        np.zeros((1, ), dtype=dtype)) is True
                elif dtype is np.void:  # pragma: no cover
                    if not fut.at_least_verion([1, 12], NP_VER):
                        with pytest.raises(TypeError) as exin:
                            fuav.is_numerical_array(
                                np.ones((1, ), dtype=dtype))
                        assert str(exin.value) == NP_VER_TYPEERROR_MSG_12
                        with pytest.raises(TypeError) as exin:
                            fuav.is_numerical_array(
                                np.zeros((1, ), dtype=dtype))
                        assert str(exin.value) == NP_VER_TYPEERROR_MSG_12
                        with pytest.raises(TypeError) as exin:
                            fuav.is_numerical_array(
                                np.empty((1, ), dtype=dtype))
                        assert str(exin.value) == NP_VER_TYPEERROR_MSG_12
                    elif not fut.at_least_verion([1, 14], NP_VER):
                        with pytest.raises(TypeError) as exin:
                            fuav.is_numerical_array(
                                np.ones((1, ), dtype=dtype))
                        assert str(exin.value) == NP_VER_TYPEERROR_MSG_14
                        assert fuav.is_numerical_array(
                            np.zeros((1, ), dtype=dtype)) is False
                        assert fuav.is_numerical_array(
                            np.empty((1, ), dtype=dtype)) is False
                    else:
                        assert fuav.is_numerical_array(
                            np.ones((1, ), dtype=dtype)) is False
                        assert fuav.is_numerical_array(
                            np.zeros((1, ), dtype=dtype)) is False
                        assert fuav.is_numerical_array(
                            np.empty((1, ), dtype=dtype)) is False
                else:
                    assert fuav.is_numerical_array(
                        np.zeros((1, ), dtype=dtype)) is False
                    assert fuav.is_numerical_array(
                        np.empty((1, ), dtype=dtype)) is False
                    assert fuav.is_numerical_array(
                        np.ones((1, ), dtype=dtype)) is False
        else:
            for dtype in dtypes:
                assert fuav.is_numerical_array(
                    np.empty((1, ), dtype=dtype)) is True
                assert fuav.is_numerical_array(
                    np.ones((1, ), dtype=dtype)) is True
                assert fuav.is_numerical_array(
                    np.zeros((1, ), dtype=dtype)) is True
        # yapf: enable


def test_is_textual_array():
    """
    Tests :func:`fatf.utils.array.validation.is_textual_array` function.
    """
    # pylint: disable=too-many-branches,too-many-statements
    type_error = 'The input should be a numpy array-like object.'
    warning_message = ('Zero-terminated bytes type is not supported and is '
                       'not considered to be a textual type. Please use any '
                       'other textual type.')
    # Test any object
    with pytest.raises(TypeError) as exin:
        fuav.is_textual_array(None)
    assert str(exin.value) == type_error

    # Test simple numerical arrays
    for i in NUMERICAL_KINDS:
        for j in [i, [i], [i] * 2, [[i] * 2] * 2]:
            array = np.array(j)
            assert fuav.is_textual_array(array) is False

    # Test simple not numerical arrays
    for i in TEXTUAL_KINDS:
        for j in [i, [i], [i] * 2, [[i] * 2] * 2]:
            array = np.array(j)
            assert fuav.is_textual_array(array) is True

    # Test simple not numerical arrays
    for i in NOT_BASE_KINDS:
        for j in [i, [i], [i] * 2, [[i] * 2] * 2]:
            array = np.array(j)
            assert fuav.is_textual_array(array) is False

    for i in UNSUPPORTED_TEXTUAL_KINDS:
        for j in [i, [i], [i] * 2, [[i] * 2] * 2]:
            array = np.array(j)
            with pytest.warns(UserWarning) as warning:
                assert fuav.is_textual_array(array) is False
            assert warning_message == str(warning[0].message)

    # Test simple numerical array
    assert fuav.is_textual_array(NUMERICAL_NP_ARRAY) is False
    # Test structured numerical array
    assert fuav.is_textual_array(NUMERICAL_STRUCTURED_ARRAY) is False
    #
    assert fuav.is_textual_array(WIDE_NP_ARRAY) is False
    assert fuav.is_textual_array(WIDE_STRUCTURED_ARRAY) is False

    # Test simple not numerical array
    assert fuav.is_textual_array(NOT_NUMERICAL_NP_ARRAY) is False
    # Test structured not numerical array
    assert fuav.is_textual_array(NOT_NUMERICAL_STRUCTURED_ARRAY) is False

    # Test base arrays
    assert fuav.is_textual_array(BASE_NP_ARRAY) is True
    assert fuav.is_textual_array(NOT_BASE_NP_ARRAY) is False
    assert fuav.is_textual_array(BASE_STRUCTURED_ARRAY) is False
    assert fuav.is_textual_array(NOT_BASE_STRUCTURED_ARRAY) is False

    # Test numpy types
    for kind, dtypes in np.sctypes.items():
        # yapf: disable
        if kind == 'others':
            for dtype in dtypes:
                if dtype is str:
                    assert fuav.is_textual_array(
                        np.empty((1, ), dtype=dtype)) is True
                    assert fuav.is_textual_array(
                        np.ones((1, ), dtype=dtype)) is True
                    assert fuav.is_textual_array(
                        np.zeros((1, ), dtype=dtype)) is True
                elif dtype is bytes:  # pragma: no cover
                    with pytest.warns(UserWarning) as warning:
                        assert fuav.is_textual_array(
                            np.zeros((1, ), dtype=dtype)) is False
                    assert warning_message == str(warning[0].message)
                    with pytest.warns(UserWarning) as warning:
                        assert fuav.is_textual_array(
                            np.empty((1, ), dtype=dtype)) is False
                    assert warning_message == str(warning[0].message)
                    with pytest.warns(UserWarning) as warning:
                        assert fuav.is_textual_array(
                            np.ones((1, ), dtype=dtype)) is False
                    assert warning_message == str(warning[0].message)
                elif dtype is np.void:  # pragma: no cover
                    if not fut.at_least_verion([1, 12], NP_VER):
                        with pytest.raises(TypeError) as exin:
                            fuav.is_textual_array(np.ones((1, ), dtype=dtype))
                        assert str(exin.value) == NP_VER_TYPEERROR_MSG_12
                        with pytest.raises(TypeError) as exin:
                            fuav.is_textual_array(
                                np.zeros((1, ), dtype=dtype))
                        assert str(exin.value) == NP_VER_TYPEERROR_MSG_12
                        with pytest.raises(TypeError) as exin:
                            fuav.is_textual_array(
                                np.empty((1, ), dtype=dtype))
                        assert str(exin.value) == NP_VER_TYPEERROR_MSG_12
                    elif not fut.at_least_verion([1, 14], NP_VER):
                        with pytest.raises(TypeError) as exin:
                            fuav.is_textual_array(np.ones((1, ), dtype=dtype))
                        assert str(exin.value) == NP_VER_TYPEERROR_MSG_14
                        assert fuav.is_textual_array(
                            np.zeros((1, ), dtype=dtype)) is False
                        assert fuav.is_textual_array(
                            np.empty((1, ), dtype=dtype)) is False
                    else:
                        assert fuav.is_textual_array(
                            np.ones((1, ), dtype=dtype)) is False
                        assert fuav.is_textual_array(
                            np.zeros((1, ), dtype=dtype)) is False
                        assert fuav.is_textual_array(
                            np.empty((1, ), dtype=dtype)) is False
                else:
                    assert fuav.is_textual_array(
                        np.zeros((1, ), dtype=dtype)) is False
                    assert fuav.is_textual_array(
                        np.empty((1, ), dtype=dtype)) is False
                    assert fuav.is_textual_array(
                        np.ones((1, ), dtype=dtype)) is False
        else:
            for dtype in dtypes:
                assert fuav.is_textual_array(
                    np.empty((1, ), dtype=dtype)) is False
                assert fuav.is_textual_array(
                    np.ones((1, ), dtype=dtype)) is False
                assert fuav.is_textual_array(
                    np.zeros((1, ), dtype=dtype)) is False
        # yapf: enable


def test_is_base_array():
    """
    Tests :func:`fatf.utils.array.validation.is_base_array` function.
    """
    # pylint: disable=too-many-branches,too-many-statements
    type_error = 'The input should be a numpy array-like object.'
    # Test any object
    with pytest.raises(TypeError) as exin:
        fuav.is_base_array(None)
    assert str(exin.value) == type_error

    # Test simple numerical arrays
    for i in BASE_KINDS:
        for j in [i, [i], [i] * 2, [[i] * 2] * 2]:
            array = np.array(j)
            assert fuav.is_base_array(array) is True

    # Test simple not numerical arrays
    for i in NOT_BASE_KINDS:
        for j in [i, [i], [i] * 2, [[i] * 2] * 2]:
            array = np.array(j)
            assert fuav.is_base_array(array) is False

    # Test simple array
    assert fuav.is_base_array(NUMERICAL_NP_ARRAY) is True
    assert fuav.is_base_array(WIDE_NP_ARRAY) is True
    assert fuav.is_base_array(NOT_NUMERICAL_NP_ARRAY) is False
    # Test structured array
    assert fuav.is_base_array(WIDE_STRUCTURED_ARRAY) is True
    assert fuav.is_base_array(NUMERICAL_STRUCTURED_ARRAY) is True
    assert fuav.is_base_array(NOT_NUMERICAL_STRUCTURED_ARRAY) is True
    # Test base arrays
    assert fuav.is_base_array(BASE_NP_ARRAY) is True
    assert fuav.is_base_array(NOT_BASE_NP_ARRAY) is False
    assert fuav.is_base_array(BASE_STRUCTURED_ARRAY) is True
    assert fuav.is_base_array(NOT_BASE_STRUCTURED_ARRAY) is False

    # Test numpy types
    for kind, dtypes in np.sctypes.items():
        # yapf: disable
        if kind == 'others':
            for dtype in dtypes:
                if dtype is bool or dtype is str or dtype is bytes:
                    assert fuav.is_base_array(
                        np.empty((1, ), dtype=dtype)) is True
                    assert fuav.is_base_array(
                        np.ones((1, ), dtype=dtype)) is True
                    assert fuav.is_base_array(
                        np.zeros((1, ), dtype=dtype)) is True
                elif dtype is np.void:  # pragma: no cover
                    if not fut.at_least_verion([1, 12], NP_VER):
                        with pytest.raises(TypeError) as exin:
                            fuav.is_base_array(np.ones((1, ), dtype=dtype))
                        assert str(exin.value) == NP_VER_TYPEERROR_MSG_12
                        with pytest.raises(TypeError) as exin:
                            fuav.is_base_array(
                                np.zeros((1, ), dtype=dtype))
                        assert str(exin.value) == NP_VER_TYPEERROR_MSG_12
                        with pytest.raises(TypeError) as exin:
                            fuav.is_base_array(
                                np.empty((1, ), dtype=dtype))
                        assert str(exin.value) == NP_VER_TYPEERROR_MSG_12
                    elif not fut.at_least_verion([1, 14], NP_VER):
                        with pytest.raises(TypeError) as exin:
                            fuav.is_base_array(np.ones((1, ), dtype=dtype))
                        assert str(exin.value) == NP_VER_TYPEERROR_MSG_14
                        assert fuav.is_base_array(
                            np.zeros((1, ), dtype=dtype)) is False
                        assert fuav.is_base_array(
                            np.empty((1, ), dtype=dtype)) is False
                    else:
                        assert fuav.is_base_array(
                            np.ones((1, ), dtype=dtype)) is False
                        assert fuav.is_base_array(
                            np.zeros((1, ), dtype=dtype)) is False
                        assert fuav.is_base_array(
                            np.empty((1, ), dtype=dtype)) is False
                else:
                    assert fuav.is_base_array(
                        np.zeros((1, ), dtype=dtype)) is False
                    assert fuav.is_base_array(
                        np.empty((1, ), dtype=dtype)) is False
                    assert fuav.is_base_array(
                        np.ones((1, ), dtype=dtype)) is False
        else:
            for dtype in dtypes:
                assert fuav.is_base_array(
                    np.empty((1, ), dtype=dtype)) is True
                assert fuav.is_base_array(
                    np.ones((1, ), dtype=dtype)) is True
                assert fuav.is_base_array(
                    np.zeros((1, ), dtype=dtype)) is True
        # yapf: enable


def test_is_2d_array():
    """
    Tests :func:`fatf.utils.array.validation.is_2d_array` function.
    """
    # pylint: disable=too-many-branches,too-many-locals,too-many-nested-blocks
    # pylint: disable=too-many-statements
    type_error = 'The input should be a numpy array-like.'
    warning_message = ('2-dimensional arrays with 1D structured elements are '
                       'not acceptable. Such a numpy array can be expressed '
                       'as a classic 2D numpy array with a desired type.')
    # Test any object
    with pytest.raises(TypeError) as exin:
        fuav.is_2d_array(None)
    assert str(exin.value) == type_error

    # Test simple numerical and not numerical arrays
    for i in NUMERICAL_KINDS + NOT_NUMERICAL_KINDS:
        for j in [[[i] * 2] * 2]:
            assert fuav.is_2d_array(np.array(j)) is True
        for j in [i, [i], [i] * 2, [[[i] * 2] * 2] * 2]:
            assert fuav.is_2d_array(np.array(j)) is False

    # Test simple and complex numerical and not numerical arrays
    assert fuav.is_2d_array(NUMERICAL_NP_ARRAY) is True
    assert fuav.is_2d_array(NOT_NUMERICAL_NP_ARRAY) is True
    assert fuav.is_2d_array(NUMERICAL_STRUCTURED_ARRAY) is True
    assert fuav.is_2d_array(NOT_NUMERICAL_STRUCTURED_ARRAY) is True
    assert fuav.is_2d_array(WIDE_NP_ARRAY) is True
    assert fuav.is_2d_array(WIDE_STRUCTURED_ARRAY) is True

    # Test simple types
    square_shapes = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 1), (2, 2)]
    not_square_shapes = [(0, ), (1, ), (2, ), (0, 0, 0), (1, 0, 0), (0, 1, 0),
                         (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1),
                         (2, 2, 2), (2, 1, 1), (2, 2, 1)]
    for _, dtypes in np.sctypes.items():
        for dtype in dtypes:
            for shape in square_shapes:
                if dtype is np.void:  # pragma: no cover
                    if not fut.at_least_verion([1, 12], NP_VER):
                        with pytest.raises(TypeError) as exin:
                            fuav.is_2d_array(np.ones(shape=shape, dtype=dtype))
                        assert str(exin.value) == NP_VER_TYPEERROR_MSG_12
                        with pytest.raises(TypeError) as exin:
                            fuav.is_2d_array(
                                np.zeros(shape=shape, dtype=dtype))
                        assert str(exin.value) == NP_VER_TYPEERROR_MSG_12
                        with pytest.raises(TypeError) as exin:
                            fuav.is_2d_array(
                                np.empty(shape=shape, dtype=dtype))
                        assert str(exin.value) == NP_VER_TYPEERROR_MSG_12
                    elif not fut.at_least_verion([1, 14], NP_VER):
                        if 0 not in shape:
                            with pytest.raises(TypeError) as exin:
                                fuav.is_2d_array(
                                    np.ones(shape=shape, dtype=dtype))
                            assert str(exin.value) == NP_VER_TYPEERROR_MSG_14
                        else:
                            ones = np.ones(shape=shape, dtype=dtype)
                            assert fuav.is_2d_array(ones) is True
                        zeros = np.zeros(shape=shape, dtype=dtype)
                        assert fuav.is_2d_array(zeros) is True
                        empty = np.empty(shape=shape, dtype=dtype)
                        assert fuav.is_2d_array(empty) is True
                    else:
                        ones = np.ones(shape=shape, dtype=dtype)
                        assert fuav.is_2d_array(ones) is True
                        zeros = np.zeros(shape=shape, dtype=dtype)
                        assert fuav.is_2d_array(zeros) is True
                        empty = np.empty(shape=shape, dtype=dtype)
                        assert fuav.is_2d_array(empty) is True
                else:
                    ones = np.ones(shape=shape, dtype=dtype)
                    assert fuav.is_2d_array(ones) is True
                    zeros = np.zeros(shape=shape, dtype=dtype)
                    assert fuav.is_2d_array(zeros) is True
                    empty = np.empty(shape=shape, dtype=dtype)
                    assert fuav.is_2d_array(empty) is True
            for shape in not_square_shapes:
                if dtype is np.void:  # pragma: no cover
                    if not fut.at_least_verion([1, 12], NP_VER):
                        with pytest.raises(TypeError) as exin:
                            fuav.is_2d_array(np.ones(shape=shape, dtype=dtype))
                        assert str(exin.value) == NP_VER_TYPEERROR_MSG_12
                        with pytest.raises(TypeError) as exin:
                            fuav.is_2d_array(
                                np.zeros(shape=shape, dtype=dtype))
                        assert str(exin.value) == NP_VER_TYPEERROR_MSG_12
                        with pytest.raises(TypeError) as exin:
                            fuav.is_2d_array(
                                np.empty(shape=shape, dtype=dtype))
                        assert str(exin.value) == NP_VER_TYPEERROR_MSG_12
                    elif not fut.at_least_verion([1, 14], NP_VER):
                        if 0 not in shape:
                            with pytest.raises(TypeError) as exin:
                                fuav.is_2d_array(
                                    np.ones(shape=shape, dtype=dtype))
                            assert str(exin.value) == NP_VER_TYPEERROR_MSG_14
                        else:
                            ones = np.ones(shape=shape, dtype=dtype)
                            assert fuav.is_2d_array(ones) is False
                        zeros = np.zeros(shape=shape, dtype=dtype)
                        assert fuav.is_2d_array(zeros) is False
                        empty = np.empty(shape=shape, dtype=dtype)
                        assert fuav.is_2d_array(empty) is False
                    else:
                        ones = np.ones(shape=shape, dtype=dtype)
                        assert fuav.is_2d_array(ones) is False
                        zeros = np.zeros(shape=shape, dtype=dtype)
                        assert fuav.is_2d_array(zeros) is False
                        empty = np.empty(shape=shape, dtype=dtype)
                        assert fuav.is_2d_array(empty) is False
                else:
                    ones = np.ones(shape=shape, dtype=dtype)
                    assert fuav.is_2d_array(ones) is False
                    zeros = np.zeros(shape=shape, dtype=dtype)
                    assert fuav.is_2d_array(zeros) is False
                    empty = np.empty(shape=shape, dtype=dtype)
                    assert fuav.is_2d_array(empty) is False

    # Complex types
    arr = np.zeros(
        3, dtype=[('x', 'f4'), ('y', np.float32), ('value', 'f4', (2, 2))])
    assert fuav.is_2d_array(arr) is False
    arr = np.ones((2, 2), dtype=arr.dtype[2])
    assert fuav.is_2d_array(arr) is False
    # yapf: disable
    not_flat_dtype = [
        NUMERICAL_STRUCTURED_ARRAY.dtype,
        NOT_NUMERICAL_STRUCTURED_ARRAY.dtype]
    flat_dtype = [
        NUMERICAL_NP_ARRAY.dtype,
        NOT_NUMERICAL_NP_ARRAY.dtype]
    flat_struct = [
        np.dtype([('n', NUMERICAL_STRUCTURED_ARRAY.dtype[0])]),
        np.dtype([('n', NUMERICAL_STRUCTURED_ARRAY.dtype[1])]),
        np.dtype([('n', NOT_NUMERICAL_STRUCTURED_ARRAY.dtype[0])]),
        np.dtype([('n', NOT_NUMERICAL_STRUCTURED_ARRAY.dtype[1])])]
    # yapf: enable
    complex_flat_shapes = [(0, ), (1, ), (2, )]
    complex_square_shapes = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 1), (2, 2)]
    complex_not_square_shapes = [(0, 0, 0), (1, 0, 0), (0, 1, 0),
                                 (0, 0, 1), (1, 1, 0), (0, 1, 1),
                                 (1, 0, 1), (1, 1, 1), (2, 2, 2),
                                 (2, 1, 1), (2, 2, 1)]  # yapf: disable
    # Structured arrays flat with multi-demnsional tuples
    for shape in complex_not_square_shapes:
        for dtype in not_flat_dtype + flat_dtype + flat_struct:
            ones = np.ones(shape=shape, dtype=dtype)
            assert fuav.is_2d_array(ones) is False
            zeros = np.zeros(shape=shape, dtype=dtype)
            assert fuav.is_2d_array(zeros) is False
            empty = np.empty(shape=shape, dtype=dtype)
            assert fuav.is_2d_array(empty) is False
    for shape in complex_square_shapes:
        for dtype in flat_dtype:
            ones = np.ones(shape=shape, dtype=dtype)
            assert fuav.is_2d_array(ones) is True
            zeros = np.zeros(shape=shape, dtype=dtype)
            assert fuav.is_2d_array(zeros) is True
            empty = np.empty(shape=shape, dtype=dtype)
            assert fuav.is_2d_array(empty) is True
    for shape in complex_square_shapes:
        for dtype in flat_struct:
            ones = np.ones(shape=shape, dtype=dtype)
            with pytest.warns(UserWarning) as warning:
                assert fuav.is_2d_array(ones) is False
            assert warning_message == str(warning[0].message)
            zeros = np.zeros(shape=shape, dtype=dtype)
            with pytest.warns(UserWarning) as warning:
                assert fuav.is_2d_array(zeros) is False
            assert warning_message == str(warning[0].message)
            empty = np.empty(shape=shape, dtype=dtype)
            with pytest.warns(UserWarning) as warning:
                assert fuav.is_2d_array(empty) is False
            assert warning_message == str(warning[0].message)
    for shape in complex_square_shapes:
        for dtype in not_flat_dtype:
            ones = np.ones(shape=shape, dtype=dtype)
            assert fuav.is_2d_array(ones) is False
            zeros = np.zeros(shape=shape, dtype=dtype)
            assert fuav.is_2d_array(zeros) is False
            empty = np.empty(shape=shape, dtype=dtype)
            assert fuav.is_2d_array(empty) is False
    for shape in complex_flat_shapes:
        for dtype in flat_dtype:
            ones = np.ones(shape=shape, dtype=dtype)
            assert fuav.is_2d_array(ones) is False
            zeros = np.zeros(shape=shape, dtype=dtype)
            assert fuav.is_2d_array(zeros) is False
            empty = np.empty(shape=shape, dtype=dtype)
            assert fuav.is_2d_array(empty) is False
    for shape in complex_flat_shapes:
        for dtype in not_flat_dtype + flat_struct:
            ones = np.ones(shape=shape, dtype=dtype)
            assert fuav.is_2d_array(ones) is True
            zeros = np.zeros(shape=shape, dtype=dtype)
            assert fuav.is_2d_array(zeros) is True
            empty = np.empty(shape=shape, dtype=dtype)
            assert fuav.is_2d_array(empty) is True


def test_is_1d_array():
    """
    Tests :func:`fatf.utils.array.validation.is_1d_array` function.
    """
    # pylint: disable=too-many-branches,too-many-locals,too-many-statements
    # pylint: disable=too-many-nested-blocks
    type_error = 'The input should be a numpy array-like.'
    warning_message = ('Structured (pseudo) 1-dimensional arrays are not '
                       'acceptable. A 1-dimensional structured numpy array '
                       'can be expressed as a classic numpy array with a '
                       'desired type.')
    # Test any object
    with pytest.raises(TypeError) as exin:
        fuav.is_1d_array(None)
    assert str(exin.value) == type_error
    # Test structured array row
    with pytest.raises(TypeError) as exin:
        fuav.is_1d_array(NUMERICAL_STRUCTURED_ARRAY[0])
    assert str(exin.value) == type_error

    # Test simple numerical and not numerical arrays
    for i in NUMERICAL_KINDS + NOT_NUMERICAL_KINDS:
        for j in [[i], [i] * 2]:
            assert fuav.is_1d_array(np.array(j)) is True
        for j in [i, [[i] * 2] * 2, [[[i] * 2] * 2] * 2]:
            assert fuav.is_1d_array(np.array(j)) is False

    # Test complex numerical and not numerical arrays
    assert fuav.is_1d_array(NUMERICAL_NP_ARRAY) is False
    assert fuav.is_1d_array(NOT_NUMERICAL_NP_ARRAY) is False
    #
    assert fuav.is_1d_array(NUMERICAL_STRUCTURED_ARRAY) is False
    assert fuav.is_1d_array(NOT_NUMERICAL_STRUCTURED_ARRAY) is False
    #
    assert fuav.is_1d_array(WIDE_NP_ARRAY) is False
    assert fuav.is_1d_array(WIDE_STRUCTURED_ARRAY) is False

    flat_shapes = [(0, ), (1, ), (2, )]
    not_flat_shapes = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 1), (2, 2),
                       (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0),
                       (0, 1, 1), (1, 0, 1), (1, 1, 1), (2, 2, 2), (2, 1, 1),
                       (2, 2, 1)]  # yapf: disable
    for _, dtypes in np.sctypes.items():
        for dtype in dtypes:
            for shape in flat_shapes:
                if dtype is np.void:  # pragma: no cover
                    if not fut.at_least_verion([1, 12], NP_VER):
                        with pytest.raises(TypeError) as exin:
                            fuav.is_1d_array(np.ones(shape=shape, dtype=dtype))
                        assert str(exin.value) == NP_VER_TYPEERROR_MSG_12
                        with pytest.raises(TypeError) as exin:
                            fuav.is_1d_array(
                                np.zeros(shape=shape, dtype=dtype))
                        assert str(exin.value) == NP_VER_TYPEERROR_MSG_12
                        with pytest.raises(TypeError) as exin:
                            fuav.is_1d_array(
                                np.empty(shape=shape, dtype=dtype))
                        assert str(exin.value) == NP_VER_TYPEERROR_MSG_12
                    elif not fut.at_least_verion([1, 14], NP_VER):
                        if 0 not in shape:
                            with pytest.raises(TypeError) as exin:
                                fuav.is_1d_array(
                                    np.ones(shape=shape, dtype=dtype))
                            assert str(exin.value) == NP_VER_TYPEERROR_MSG_14
                        else:
                            ones = np.ones(shape=shape, dtype=dtype)
                            assert fuav.is_1d_array(ones) is True
                        zeros = np.zeros(shape=shape, dtype=dtype)
                        assert fuav.is_1d_array(zeros) is True
                        empty = np.empty(shape=shape, dtype=dtype)
                        assert fuav.is_1d_array(empty) is True
                    else:
                        ones = np.ones(shape=shape, dtype=dtype)
                        assert fuav.is_1d_array(ones) is True
                        zeros = np.zeros(shape=shape, dtype=dtype)
                        assert fuav.is_1d_array(zeros) is True
                        empty = np.empty(shape=shape, dtype=dtype)
                        assert fuav.is_1d_array(empty) is True
                else:
                    ones = np.ones(shape=shape, dtype=dtype)
                    assert fuav.is_1d_array(ones) is True
                    zeros = np.zeros(shape=shape, dtype=dtype)
                    assert fuav.is_1d_array(zeros) is True
                    empty = np.empty(shape=shape, dtype=dtype)
                    assert fuav.is_1d_array(empty) is True
            for shape in not_flat_shapes:
                if dtype is np.void:  # pragma: no cover
                    if not fut.at_least_verion([1, 12], NP_VER):
                        with pytest.raises(TypeError) as exin:
                            fuav.is_1d_array(np.ones(shape=shape, dtype=dtype))
                        assert str(exin.value) == NP_VER_TYPEERROR_MSG_12
                        with pytest.raises(TypeError) as exin:
                            fuav.is_1d_array(
                                np.zeros(shape=shape, dtype=dtype))
                        assert str(exin.value) == NP_VER_TYPEERROR_MSG_12
                        with pytest.raises(TypeError) as exin:
                            fuav.is_1d_array(
                                np.empty(shape=shape, dtype=dtype))
                        assert str(exin.value) == NP_VER_TYPEERROR_MSG_12
                    elif not fut.at_least_verion([1, 14], NP_VER):
                        if 0 not in shape:
                            with pytest.raises(TypeError) as exin:
                                fuav.is_1d_array(
                                    np.ones(shape=shape, dtype=dtype))
                            assert str(exin.value) == NP_VER_TYPEERROR_MSG_14
                        else:
                            ones = np.ones(shape=shape, dtype=dtype)
                            assert fuav.is_1d_array(ones) is False
                        zeros = np.zeros(shape=shape, dtype=dtype)
                        assert fuav.is_1d_array(zeros) is False
                        empty = np.empty(shape=shape, dtype=dtype)
                        assert fuav.is_1d_array(empty) is False
                    else:
                        ones = np.ones(shape=shape, dtype=dtype)
                        assert fuav.is_1d_array(ones) is False
                        zeros = np.zeros(shape=shape, dtype=dtype)
                        assert fuav.is_1d_array(zeros) is False
                        empty = np.empty(shape=shape, dtype=dtype)
                        assert fuav.is_1d_array(empty) is False
                else:
                    ones = np.ones(shape=shape, dtype=dtype)
                    assert fuav.is_1d_array(ones) is False
                    zeros = np.zeros(shape=shape, dtype=dtype)
                    assert fuav.is_1d_array(zeros) is False
                    empty = np.empty(shape=shape, dtype=dtype)
                    assert fuav.is_1d_array(empty) is False

    # yapf: disable
    not_flat_dtype = [
        NUMERICAL_STRUCTURED_ARRAY.dtype,
        NOT_NUMERICAL_STRUCTURED_ARRAY.dtype]
    flat_dtype = [
        NUMERICAL_NP_ARRAY.dtype,
        NOT_NUMERICAL_NP_ARRAY.dtype]
    flat_struct = [
        np.dtype([('n', NUMERICAL_STRUCTURED_ARRAY.dtype[0])]),
        np.dtype([('n', NUMERICAL_STRUCTURED_ARRAY.dtype[1])]),
        np.dtype([('n', NOT_NUMERICAL_STRUCTURED_ARRAY.dtype[0])]),
        np.dtype([('n', NOT_NUMERICAL_STRUCTURED_ARRAY.dtype[1])])]
    # yapf: enable
    for shape in flat_shapes:
        for dtype in flat_dtype:
            ones = np.ones(shape=shape, dtype=dtype)
            assert fuav.is_1d_array(ones) is True
            zeros = np.zeros(shape=shape, dtype=dtype)
            assert fuav.is_1d_array(zeros) is True
            empty = np.empty(shape=shape, dtype=dtype)
            assert fuav.is_1d_array(empty) is True
    for shape in flat_shapes:
        for dtype in flat_struct:
            ones = np.ones(shape=shape, dtype=dtype)
            with pytest.warns(UserWarning) as warning:
                assert fuav.is_1d_array(ones) is False
            assert warning_message == str(warning[0].message)
            #
            zeros = np.zeros(shape=shape, dtype=dtype)
            with pytest.warns(UserWarning) as warning:
                assert fuav.is_1d_array(zeros) is False
            assert warning_message == str(warning[0].message)
            #
            empty = np.empty(shape=shape, dtype=dtype)
            with pytest.warns(UserWarning) as warning:
                assert fuav.is_1d_array(empty) is False
            assert warning_message == str(warning[0].message)
    for shape in flat_shapes:
        for dtype in not_flat_dtype:
            ones = np.ones(shape=shape, dtype=dtype)
            assert fuav.is_1d_array(ones) is False
            zeros = np.zeros(shape=shape, dtype=dtype)
            assert fuav.is_1d_array(zeros) is False
            empty = np.empty(shape=shape, dtype=dtype)
            assert fuav.is_1d_array(empty) is False
    for shape in not_flat_shapes:
        for dtype in not_flat_dtype + flat_dtype + flat_struct:
            ones = np.ones(shape=shape, dtype=dtype)
            assert fuav.is_1d_array(ones) is False
            zeros = np.zeros(shape=shape, dtype=dtype)
            assert fuav.is_1d_array(zeros) is False
            empty = np.empty(shape=shape, dtype=dtype)
            assert fuav.is_1d_array(empty) is False


def test_is_structured_row():
    """
    Tests :func:`fatf.utils.array.validation.is_structured_row` function.
    """
    type_error = ('The input should be a row of a structured numpy array '
                  '(numpy.void type).')
    # Wrong type
    with pytest.raises(TypeError) as exin:
        fuav.is_structured_row(None)
    assert str(exin.value) == type_error
    # Simple arrays
    with pytest.raises(TypeError) as exin:
        fuav.is_structured_row(np.ones((7, 15), dtype=float))
    assert str(exin.value) == type_error
    with pytest.raises(TypeError) as exin:
        fuav.is_structured_row(np.ones((4, ), dtype=float))
    assert str(exin.value) == type_error
    # Structured arrays
    with pytest.raises(TypeError) as exin:
        fuav.is_structured_row(NUMERICAL_STRUCTURED_ARRAY)
    assert str(exin.value) == type_error
    # Structured 0-dimensional arrays
    with pytest.raises(TypeError) as exin:
        fuav.is_structured_row(
            np.array((1., (1 + 1j)), dtype=[('n', '<f8'), ('c', '<c16')]))
    assert str(exin.value) == type_error
    # Structured 1-dimensional arrays
    with pytest.raises(TypeError) as exin:
        fuav.is_structured_row(
            np.array([(1., (1 + 1j))], dtype=[('n', '<f8'), ('c', '<c16')]))
    assert str(exin.value) == type_error
    with pytest.raises(TypeError) as exin:
        fuav.is_structured_row(
            np.array([(1., ), (2, ), (3, )], dtype=[('n', '<f8')]))
    assert str(exin.value) == type_error
    # Void types
    void_array = np.array([b'123'], np.void)
    with pytest.raises(TypeError) as exin:
        fuav.is_structured_row(void_array)
    assert str(exin.value) == type_error
    assert not fuav.is_structured_row(void_array[0])
    void_array = np.array([b'123', b'888'], np.void)
    with pytest.raises(TypeError) as exin:
        fuav.is_structured_row(void_array)
    assert str(exin.value) == type_error
    assert not fuav.is_structured_row(void_array[1])
    # Structured rows
    assert fuav.is_structured_row(NUMERICAL_STRUCTURED_ARRAY[0])
    assert fuav.is_structured_row(NOT_NUMERICAL_STRUCTURED_ARRAY[1])
    assert fuav.is_structured_row(BASE_STRUCTURED_ARRAY[2])
    assert fuav.is_structured_row(NOT_BASE_STRUCTURED_ARRAY[3])


def test_is_1d_like():
    """
    Tests :func:`fatf.utils.array.validation.is_1d_like` function.
    """
    type_error = ('The input should either be a numpy array-like object '
                  '(numpy.ndarray) or a row of a structured numpy array '
                  '(numpy.void).')
    # None type
    with pytest.raises(TypeError) as exin:
        fuav.is_1d_like(None)
    assert str(exin.value) == type_error
    # Array 2D
    assert not fuav.is_1d_like(np.ones((42, 24), dtype=float))
    # Array 1D
    assert fuav.is_1d_like(np.ones((42, ), dtype=float))
    # Structured 2D
    assert not fuav.is_1d_like(NUMERICAL_STRUCTURED_ARRAY)
    assert not fuav.is_1d_like(NOT_NUMERICAL_STRUCTURED_ARRAY)
    assert not fuav.is_1d_like(BASE_STRUCTURED_ARRAY)
    assert not fuav.is_1d_like(NOT_BASE_STRUCTURED_ARRAY)
    # Structured 1D
    assert not fuav.is_1d_like(
        np.array([(1., (1 + 1j))], dtype=[('n', '<f8'), ('c', '<c16')]))
    user_warning = ('Structured (pseudo) 1-dimensional arrays are not '
                    'acceptable. A 1-dimensional structured numpy array can '
                    'be expressed as a classic numpy array with a desired '
                    'type.')
    with pytest.warns(UserWarning) as warning:
        assert not fuav.is_1d_like(
            np.array([(1., ), (2, ), (3, )], dtype=[('n', '<f8')]))
    assert str(warning[0].message) == user_warning
    # Structured row
    assert fuav.is_1d_like(NUMERICAL_STRUCTURED_ARRAY[0])
    assert fuav.is_1d_like(NOT_NUMERICAL_STRUCTURED_ARRAY[1])
    assert fuav.is_1d_like(BASE_STRUCTURED_ARRAY[2])
    assert fuav.is_1d_like(NOT_BASE_STRUCTURED_ARRAY[3])
    # Numpy void
    void_array = np.array([b'123'], np.void)
    assert fuav.is_1d_like(void_array)
    assert not fuav.is_1d_like(void_array[0])
    void_array = np.array([b'123', b'888'], np.void)
    assert fuav.is_1d_like(void_array)
    assert not fuav.is_1d_like(void_array[1])


def test_is_structured_array():
    """
    Tests :func:`fatf.utils.array.validation.is_structured_array` function.
    """
    type_error = 'The input should be a numpy array-like.'
    # Test any object
    with pytest.raises(TypeError) as exin:
        fuav.is_structured_array(None)
    assert str(exin.value) == type_error

    assert fuav.is_structured_array(NUMERICAL_NP_ARRAY) is False
    assert fuav.is_structured_array(NOT_NUMERICAL_NP_ARRAY) is False
    assert fuav.is_structured_array(WIDE_NP_ARRAY) is False
    assert fuav.is_structured_array(NUMERICAL_STRUCTURED_ARRAY) is True
    assert fuav.is_structured_array(NOT_NUMERICAL_STRUCTURED_ARRAY) is True
    assert fuav.is_structured_array(WIDE_STRUCTURED_ARRAY) is True

    shapes = [(0, ), (1, ), (2, ), (0, 0), (0, 1), (1, 0), (1, 1), (2, 1),
              (2, 2), (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0),
              (0, 1, 1), (1, 0, 1), (1, 1, 1), (2, 2, 2), (2, 1, 1), (2, 2, 1)]
    # yapf: disable
    basic_dtype = [
        NUMERICAL_NP_ARRAY.dtype,
        NOT_NUMERICAL_NP_ARRAY.dtype,
        NUMERICAL_STRUCTURED_ARRAY.dtype[0],
        NUMERICAL_STRUCTURED_ARRAY.dtype[1],
        NOT_NUMERICAL_STRUCTURED_ARRAY.dtype[0],
        NOT_NUMERICAL_STRUCTURED_ARRAY.dtype[1]]
    struct_dtype = [
        NUMERICAL_STRUCTURED_ARRAY.dtype,
        NOT_NUMERICAL_STRUCTURED_ARRAY.dtype,
        np.dtype([('n', NUMERICAL_STRUCTURED_ARRAY.dtype[0])]),
        np.dtype([('n', NUMERICAL_STRUCTURED_ARRAY.dtype[1])]),
        np.dtype([('n', NOT_NUMERICAL_STRUCTURED_ARRAY.dtype[0])]),
        np.dtype([('n', NOT_NUMERICAL_STRUCTURED_ARRAY.dtype[1])])]
    # yapf: enable
    for shape in shapes:
        for dtype in basic_dtype:
            ones = np.ones(shape=shape, dtype=dtype)
            assert fuav.is_structured_array(ones) is False
            zeros = np.zeros(shape=shape, dtype=dtype)
            assert fuav.is_structured_array(zeros) is False
            empty = np.empty(shape=shape, dtype=dtype)
            assert fuav.is_structured_array(empty) is False
        for dtype in struct_dtype:
            ones = np.ones(shape=shape, dtype=dtype)
            assert fuav.is_structured_array(ones) is True
            zeros = np.zeros(shape=shape, dtype=dtype)
            assert fuav.is_structured_array(zeros) is True
            empty = np.empty(shape=shape, dtype=dtype)
            assert fuav.is_structured_array(empty) is True
