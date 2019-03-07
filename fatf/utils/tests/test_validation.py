"""
Tests functions responsible for objects validation across FAT-Forensics.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import numpy as np
import pytest

import fatf.exceptions
import fatf.utils.validation as fuv

NUMERICAL_KINDS = [True, 1, -1, 1.0, 1 + 1j]
NOT_NUMERICAL_KINDS = [object(), 'string', u'unicode', None]

NUMERICAL_NP_ARRAY = np.array([
    [True, 1],
    [-1, 1.0],
    [1 + 1j, False]])  # yapf: disable
NOT_NUMERICAL_NP_ARRAY = np.array([
    [True, 1],
    [-1, 1.0],
    [1 + 1j, False],
    ['a', 'b']])  # yapf: disable
WIDE_NP_ARRAY = np.array([
    [True, 1, 0],
    [-1, 1.0, 4],
    [1 + 1j, False, 2]])  # yapf: disable
NUMERICAL_STRUCTURED_ARRAY = np.array([
    (1.0, 1.0 + 1j),
    (1, 1 + 1j),
    (-1, -1 + 1j)], dtype=[('numbers', '<f8'),
                           ('complex', '<c16')])  # yapf: disable
NOT_NUMERICAL_STRUCTURED_ARRAY = np.array([
    (True, 'a'),
    (1, 'b'),
    (-1, 'c'),
    (1.0, 'd'),
    (1 + 1j, 'e'),
    (False, 'f')], dtype=[('numerical', 'c8'),
                          ('categorical', 'U1')])  # yapf: disable
WIDE_STRUCTURED_ARRAY = np.array([
    (1.0, 1.0 + 1j, 6),
    (1, 1 + 1j, 6),
    (-1, -1 + 1j, 6)], dtype=[('numbers', '<f8'),
                              ('complex', '<c16'),
                              ('anybody', '<f8')])  # yapf: disable

NP_VER = [int(i) for i in np.version.version.split('.')]
NP_VER_TYPEERROR_MSG_14 = 'a bytes-like object is required, not \'int\''
NP_VER_TYPEERROR_MSG_12 = 'Empty data-type'


def test_is_numerical_dtype():
    """
    Tests :func:`fatf.utils.validation.is_numerical_dtype` function.
    """
    type_error = 'The input should be a numpy dtype object.'
    value_error = ('The numpy dtype object is structured. '
                   'Only base dtype are allowed.')
    # Test any object
    with pytest.raises(TypeError) as exin:
        fuv.is_numerical_dtype(None)
    assert str(exin.value) == type_error

    # Test simple numerical arrays
    for i in NUMERICAL_KINDS:
        for j in [i, [i], [i] * 2, [[i] * 2] * 2]:
            array = np.array(j)
            array_dtype = array.dtype
            assert fuv.is_numerical_dtype(array_dtype) is True

    # Test simple not numerical arrays
    for i in NOT_NUMERICAL_KINDS:
        for j in [i, [i], [i] * 2, [[i] * 2] * 2]:
            array = np.array(j)
            array_dtype = array.dtype
            assert fuv.is_numerical_dtype(array_dtype) is False

    # Test simple numerical array
    assert fuv.is_numerical_dtype(NUMERICAL_NP_ARRAY.dtype) is True
    # Test simple not numerical array
    assert fuv.is_numerical_dtype(NOT_NUMERICAL_NP_ARRAY.dtype) is False

    # Test structured numerical array
    with pytest.raises(ValueError) as exin:
        fuv.is_numerical_dtype(NUMERICAL_STRUCTURED_ARRAY.dtype)
    assert str(exin.value) == value_error
    # Test structured not numerical array
    with pytest.raises(ValueError) as exin:
        fuv.is_numerical_dtype(NOT_NUMERICAL_STRUCTURED_ARRAY.dtype)
    assert str(exin.value) == value_error

    # Test numpy types
    for kind, dtypes in np.sctypes.items():
        if kind == 'others':
            for dtype in dtypes:
                if dtype is bool:
                    assert fuv.is_numerical_dtype(np.dtype(dtype)) is True
                else:
                    assert fuv.is_numerical_dtype(np.dtype(dtype)) is False
        else:
            for dtype in dtypes:
                assert fuv.is_numerical_dtype(np.dtype(dtype)) is True


def test_is_numerical_array():
    """
    Tests :func:`fatf.utils.validation.is_numerical_array` function.
    """
    # pylint: disable=too-many-branches
    type_error = 'The input should be a numpy array-like object.'
    # Test any object
    with pytest.raises(TypeError) as exin:
        fuv.is_numerical_array(None)
    assert str(exin.value) == type_error

    # Test simple numerical arrays
    for i in NUMERICAL_KINDS:
        for j in [i, [i], [i] * 2, [[i] * 2] * 2]:
            array = np.array(j)
            assert fuv.is_numerical_array(array) is True

    # Test simple not numerical arrays
    for i in NOT_NUMERICAL_KINDS:
        for j in [i, [i], [i] * 2, [[i] * 2] * 2]:
            array = np.array(j)
            assert fuv.is_numerical_array(array) is False

    # Test simple numerical array
    assert fuv.is_numerical_array(NUMERICAL_NP_ARRAY) is True
    # Test structured numerical array
    assert fuv.is_numerical_array(NUMERICAL_STRUCTURED_ARRAY) is True
    #
    assert fuv.is_numerical_array(WIDE_NP_ARRAY) is True
    assert fuv.is_numerical_array(WIDE_STRUCTURED_ARRAY) is True

    # Test simple not numerical array
    assert fuv.is_numerical_array(NOT_NUMERICAL_NP_ARRAY) is False
    # Test structured not numerical array
    assert fuv.is_numerical_array(NOT_NUMERICAL_STRUCTURED_ARRAY) is False

    # Test numpy types
    for kind, dtypes in np.sctypes.items():
        # yapf: disable
        if kind == 'others':
            for dtype in dtypes:
                if dtype is bool:
                    assert fuv.is_numerical_array(
                        np.empty((1, ), dtype=dtype)) is True
                    assert fuv.is_numerical_array(
                        np.ones((1, ), dtype=dtype)) is True
                    assert fuv.is_numerical_array(
                        np.zeros((1, ), dtype=dtype)) is True
                elif dtype is np.void:  # pragma: no cover
                    if NP_VER[0] < 2 and NP_VER[1] < 12:
                        with pytest.raises(TypeError) as exin:
                            fuv.is_numerical_array(np.ones((1, ), dtype=dtype))
                        assert str(exin.value) == NP_VER_TYPEERROR_MSG_12
                        with pytest.raises(TypeError) as exin:
                            fuv.is_numerical_array(
                                np.zeros((1, ), dtype=dtype))
                        assert str(exin.value) == NP_VER_TYPEERROR_MSG_12
                        with pytest.raises(TypeError) as exin:
                            fuv.is_numerical_array(
                                np.empty((1, ), dtype=dtype))
                        assert str(exin.value) == NP_VER_TYPEERROR_MSG_12
                    elif NP_VER[0] < 2 and NP_VER[1] < 14:
                        with pytest.raises(TypeError) as exin:
                            fuv.is_numerical_array(np.ones((1, ), dtype=dtype))
                        assert str(exin.value) == NP_VER_TYPEERROR_MSG_14
                        assert fuv.is_numerical_array(
                            np.zeros((1, ), dtype=dtype)) is False
                        assert fuv.is_numerical_array(
                            np.empty((1, ), dtype=dtype)) is False
                    else:
                        assert fuv.is_numerical_array(
                            np.ones((1, ), dtype=dtype)) is False
                        assert fuv.is_numerical_array(
                            np.zeros((1, ), dtype=dtype)) is False
                        assert fuv.is_numerical_array(
                            np.empty((1, ), dtype=dtype)) is False
                else:
                    assert fuv.is_numerical_array(
                        np.zeros((1, ), dtype=dtype)) is False
                    assert fuv.is_numerical_array(
                        np.empty((1, ), dtype=dtype)) is False
                    assert fuv.is_numerical_array(
                        np.ones((1, ), dtype=dtype)) is False
        else:
            for dtype in dtypes:
                assert fuv.is_numerical_array(
                    np.empty((1, ), dtype=dtype)) is True
                assert fuv.is_numerical_array(
                    np.ones((1, ), dtype=dtype)) is True
                assert fuv.is_numerical_array(
                    np.zeros((1, ), dtype=dtype)) is True
        # yapf: enable


def test_is_2d_array():
    """
    Tests :func:`fatf.utils.validation.is_2d_array` function.
    """
    # pylint: disable=too-many-branches,too-many-locals,too-many-nested-blocks
    # pylint: disable=too-many-statements
    type_error = 'The input should be a numpy array-like.'
    warning_message = ('2-dimensional arrays with 1D structured elements are '
                       'not acceptable. Such a numpy array can be expressed '
                       'as a classic 2D numpy array with a desired type.')
    # Test any object
    with pytest.raises(TypeError) as exin:
        fuv.is_2d_array(None)
    assert str(exin.value) == type_error

    # Test simple numerical and not numerical arrays
    for i in NUMERICAL_KINDS + NOT_NUMERICAL_KINDS:
        for j in [[[i] * 2] * 2]:
            assert fuv.is_2d_array(np.array(j)) is True
        for j in [i, [i], [i] * 2, [[[i] * 2] * 2] * 2]:
            assert fuv.is_2d_array(np.array(j)) is False

    # Test simple and complex numerical and not numerical arrays
    assert fuv.is_2d_array(NUMERICAL_NP_ARRAY) is True
    assert fuv.is_2d_array(NOT_NUMERICAL_NP_ARRAY) is True
    assert fuv.is_2d_array(NUMERICAL_STRUCTURED_ARRAY) is True
    assert fuv.is_2d_array(NOT_NUMERICAL_STRUCTURED_ARRAY) is True
    assert fuv.is_2d_array(WIDE_NP_ARRAY) is True
    assert fuv.is_2d_array(WIDE_STRUCTURED_ARRAY) is True

    # Test simple types
    square_shapes = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 1), (2, 2)]
    not_square_shapes = [(0, ), (1, ), (2, ), (0, 0, 0), (1, 0, 0), (0, 1, 0),
                         (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1),
                         (2, 2, 2), (2, 1, 1), (2, 2, 1)]
    for _, dtypes in np.sctypes.items():
        for dtype in dtypes:
            for shape in square_shapes:
                if dtype is np.void:  # pragma: no cover
                    if NP_VER[0] < 2 and NP_VER[1] < 12:
                        with pytest.raises(TypeError) as exin:
                            fuv.is_2d_array(np.ones(shape=shape, dtype=dtype))
                        assert str(exin.value) == NP_VER_TYPEERROR_MSG_12
                        with pytest.raises(TypeError) as exin:
                            fuv.is_2d_array(np.zeros(shape=shape, dtype=dtype))
                        assert str(exin.value) == NP_VER_TYPEERROR_MSG_12
                        with pytest.raises(TypeError) as exin:
                            fuv.is_2d_array(np.empty(shape=shape, dtype=dtype))
                        assert str(exin.value) == NP_VER_TYPEERROR_MSG_12
                    elif NP_VER[0] < 2 and NP_VER[1] < 14:
                        if 0 not in shape:
                            with pytest.raises(TypeError) as exin:
                                fuv.is_2d_array(
                                    np.ones(shape=shape, dtype=dtype))
                            assert str(exin.value) == NP_VER_TYPEERROR_MSG_14
                        else:
                            ones = np.ones(shape=shape, dtype=dtype)
                            assert fuv.is_2d_array(ones) is True
                        zeros = np.zeros(shape=shape, dtype=dtype)
                        assert fuv.is_2d_array(zeros) is True
                        empty = np.empty(shape=shape, dtype=dtype)
                        assert fuv.is_2d_array(empty) is True
                    else:
                        ones = np.ones(shape=shape, dtype=dtype)
                        assert fuv.is_2d_array(ones) is True
                        zeros = np.zeros(shape=shape, dtype=dtype)
                        assert fuv.is_2d_array(zeros) is True
                        empty = np.empty(shape=shape, dtype=dtype)
                        assert fuv.is_2d_array(empty) is True
                else:
                    ones = np.ones(shape=shape, dtype=dtype)
                    assert fuv.is_2d_array(ones) is True
                    zeros = np.zeros(shape=shape, dtype=dtype)
                    assert fuv.is_2d_array(zeros) is True
                    empty = np.empty(shape=shape, dtype=dtype)
                    assert fuv.is_2d_array(empty) is True
            for shape in not_square_shapes:
                if dtype is np.void:  # pragma: no cover
                    if NP_VER[0] < 2 and NP_VER[1] < 12:
                        with pytest.raises(TypeError) as exin:
                            fuv.is_2d_array(np.ones(shape=shape, dtype=dtype))
                        assert str(exin.value) == NP_VER_TYPEERROR_MSG_12
                        with pytest.raises(TypeError) as exin:
                            fuv.is_2d_array(np.zeros(shape=shape, dtype=dtype))
                        assert str(exin.value) == NP_VER_TYPEERROR_MSG_12
                        with pytest.raises(TypeError) as exin:
                            fuv.is_2d_array(np.empty(shape=shape, dtype=dtype))
                        assert str(exin.value) == NP_VER_TYPEERROR_MSG_12
                    elif NP_VER[0] < 2 and NP_VER[1] < 14:
                        if 0 not in shape:
                            with pytest.raises(TypeError) as exin:
                                fuv.is_2d_array(
                                    np.ones(shape=shape, dtype=dtype))
                            assert str(exin.value) == NP_VER_TYPEERROR_MSG_14
                        else:
                            ones = np.ones(shape=shape, dtype=dtype)
                            assert fuv.is_2d_array(ones) is False
                        zeros = np.zeros(shape=shape, dtype=dtype)
                        assert fuv.is_2d_array(zeros) is False
                        empty = np.empty(shape=shape, dtype=dtype)
                        assert fuv.is_2d_array(empty) is False
                    else:
                        ones = np.ones(shape=shape, dtype=dtype)
                        assert fuv.is_2d_array(ones) is False
                        zeros = np.zeros(shape=shape, dtype=dtype)
                        assert fuv.is_2d_array(zeros) is False
                        empty = np.empty(shape=shape, dtype=dtype)
                        assert fuv.is_2d_array(empty) is False
                else:
                    ones = np.ones(shape=shape, dtype=dtype)
                    assert fuv.is_2d_array(ones) is False
                    zeros = np.zeros(shape=shape, dtype=dtype)
                    assert fuv.is_2d_array(zeros) is False
                    empty = np.empty(shape=shape, dtype=dtype)
                    assert fuv.is_2d_array(empty) is False

    # Complex types
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
            assert fuv.is_2d_array(ones) is False
            zeros = np.zeros(shape=shape, dtype=dtype)
            assert fuv.is_2d_array(zeros) is False
            empty = np.empty(shape=shape, dtype=dtype)
            assert fuv.is_2d_array(empty) is False
    for shape in complex_square_shapes:
        for dtype in flat_dtype:
            ones = np.ones(shape=shape, dtype=dtype)
            assert fuv.is_2d_array(ones) is True
            zeros = np.zeros(shape=shape, dtype=dtype)
            assert fuv.is_2d_array(zeros) is True
            empty = np.empty(shape=shape, dtype=dtype)
            assert fuv.is_2d_array(empty) is True
    for shape in complex_square_shapes:
        for dtype in flat_struct:
            ones = np.ones(shape=shape, dtype=dtype)
            with pytest.warns(UserWarning) as warning:
                assert fuv.is_2d_array(ones) is False
            assert warning_message == str(warning[0].message)
            zeros = np.zeros(shape=shape, dtype=dtype)
            with pytest.warns(UserWarning) as warning:
                assert fuv.is_2d_array(zeros) is False
            assert warning_message == str(warning[0].message)
            empty = np.empty(shape=shape, dtype=dtype)
            with pytest.warns(UserWarning) as warning:
                assert fuv.is_2d_array(empty) is False
            assert warning_message == str(warning[0].message)
    for shape in complex_square_shapes:
        for dtype in not_flat_dtype:
            ones = np.ones(shape=shape, dtype=dtype)
            assert fuv.is_2d_array(ones) is False
            zeros = np.zeros(shape=shape, dtype=dtype)
            assert fuv.is_2d_array(zeros) is False
            empty = np.empty(shape=shape, dtype=dtype)
            assert fuv.is_2d_array(empty) is False
    for shape in complex_flat_shapes:
        for dtype in flat_dtype + flat_struct:
            ones = np.ones(shape=shape, dtype=dtype)
            assert fuv.is_2d_array(ones) is False
            zeros = np.zeros(shape=shape, dtype=dtype)
            assert fuv.is_2d_array(zeros) is False
            empty = np.empty(shape=shape, dtype=dtype)
            assert fuv.is_2d_array(empty) is False
    for shape in complex_flat_shapes:
        for dtype in not_flat_dtype:
            ones = np.ones(shape=shape, dtype=dtype)
            assert fuv.is_2d_array(ones) is True
            zeros = np.zeros(shape=shape, dtype=dtype)
            assert fuv.is_2d_array(zeros) is True
            empty = np.empty(shape=shape, dtype=dtype)
            assert fuv.is_2d_array(empty) is True


def test_is_1d_array():
    """
    Tests :func:`fatf.utils.validation.is_1d_array` function.
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
        fuv.is_1d_array(None)
    assert str(exin.value) == type_error

    # Test simple numerical and not numerical arrays
    for i in NUMERICAL_KINDS + NOT_NUMERICAL_KINDS:
        for j in [[i], [i] * 2]:
            assert fuv.is_1d_array(np.array(j)) is True
        for j in [i, [[i] * 2] * 2, [[[i] * 2] * 2] * 2]:
            assert fuv.is_1d_array(np.array(j)) is False

    # Test complex numerical and not numerical arrays
    assert fuv.is_1d_array(NUMERICAL_NP_ARRAY) is False
    assert fuv.is_1d_array(NOT_NUMERICAL_NP_ARRAY) is False
    #
    assert fuv.is_1d_array(NUMERICAL_STRUCTURED_ARRAY) is False
    assert fuv.is_1d_array(NOT_NUMERICAL_STRUCTURED_ARRAY) is False
    #
    assert fuv.is_1d_array(WIDE_NP_ARRAY) is False
    assert fuv.is_1d_array(WIDE_STRUCTURED_ARRAY) is False

    flat_shapes = [(0, ), (1, ), (2, )]
    not_flat_shapes = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 1), (2, 2),
                       (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0),
                       (0, 1, 1), (1, 0, 1), (1, 1, 1), (2, 2, 2), (2, 1, 1),
                       (2, 2, 1)]  # yapf: disable
    for _, dtypes in np.sctypes.items():
        for dtype in dtypes:
            for shape in flat_shapes:
                if dtype is np.void:  # pragma: no cover
                    if NP_VER[0] < 2 and NP_VER[1] < 12:
                        with pytest.raises(TypeError) as exin:
                            fuv.is_1d_array(np.ones(shape=shape, dtype=dtype))
                        assert str(exin.value) == NP_VER_TYPEERROR_MSG_12
                        with pytest.raises(TypeError) as exin:
                            fuv.is_1d_array(np.zeros(shape=shape, dtype=dtype))
                        assert str(exin.value) == NP_VER_TYPEERROR_MSG_12
                        with pytest.raises(TypeError) as exin:
                            fuv.is_1d_array(np.empty(shape=shape, dtype=dtype))
                        assert str(exin.value) == NP_VER_TYPEERROR_MSG_12
                    elif NP_VER[0] < 2 and NP_VER[1] < 14:
                        if 0 not in shape:
                            with pytest.raises(TypeError) as exin:
                                fuv.is_1d_array(
                                    np.ones(shape=shape, dtype=dtype))
                            assert str(exin.value) == NP_VER_TYPEERROR_MSG_14
                        else:
                            ones = np.ones(shape=shape, dtype=dtype)
                            assert fuv.is_1d_array(ones) is True
                        zeros = np.zeros(shape=shape, dtype=dtype)
                        assert fuv.is_1d_array(zeros) is True
                        empty = np.empty(shape=shape, dtype=dtype)
                        assert fuv.is_1d_array(empty) is True
                    else:
                        ones = np.ones(shape=shape, dtype=dtype)
                        assert fuv.is_1d_array(ones) is True
                        zeros = np.zeros(shape=shape, dtype=dtype)
                        assert fuv.is_1d_array(zeros) is True
                        empty = np.empty(shape=shape, dtype=dtype)
                        assert fuv.is_1d_array(empty) is True
                else:
                    ones = np.ones(shape=shape, dtype=dtype)
                    assert fuv.is_1d_array(ones) is True
                    zeros = np.zeros(shape=shape, dtype=dtype)
                    assert fuv.is_1d_array(zeros) is True
                    empty = np.empty(shape=shape, dtype=dtype)
                    assert fuv.is_1d_array(empty) is True
            for shape in not_flat_shapes:
                if dtype is np.void:  # pragma: no cover
                    if NP_VER[0] < 2 and NP_VER[1] < 12:
                        with pytest.raises(TypeError) as exin:
                            fuv.is_1d_array(np.ones(shape=shape, dtype=dtype))
                        assert str(exin.value) == NP_VER_TYPEERROR_MSG_12
                        with pytest.raises(TypeError) as exin:
                            fuv.is_1d_array(np.zeros(shape=shape, dtype=dtype))
                        assert str(exin.value) == NP_VER_TYPEERROR_MSG_12
                        with pytest.raises(TypeError) as exin:
                            fuv.is_1d_array(np.empty(shape=shape, dtype=dtype))
                        assert str(exin.value) == NP_VER_TYPEERROR_MSG_12
                    elif NP_VER[0] < 2 and NP_VER[1] < 14:
                        if 0 not in shape:
                            with pytest.raises(TypeError) as exin:
                                fuv.is_1d_array(
                                    np.ones(shape=shape, dtype=dtype))
                            assert str(exin.value) == NP_VER_TYPEERROR_MSG_14
                        else:
                            ones = np.ones(shape=shape, dtype=dtype)
                            assert fuv.is_1d_array(ones) is False
                        zeros = np.zeros(shape=shape, dtype=dtype)
                        assert fuv.is_1d_array(zeros) is False
                        empty = np.empty(shape=shape, dtype=dtype)
                        assert fuv.is_1d_array(empty) is False
                    else:
                        ones = np.ones(shape=shape, dtype=dtype)
                        assert fuv.is_1d_array(ones) is False
                        zeros = np.zeros(shape=shape, dtype=dtype)
                        assert fuv.is_1d_array(zeros) is False
                        empty = np.empty(shape=shape, dtype=dtype)
                        assert fuv.is_1d_array(empty) is False
                else:
                    ones = np.ones(shape=shape, dtype=dtype)
                    assert fuv.is_1d_array(ones) is False
                    zeros = np.zeros(shape=shape, dtype=dtype)
                    assert fuv.is_1d_array(zeros) is False
                    empty = np.empty(shape=shape, dtype=dtype)
                    assert fuv.is_1d_array(empty) is False

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
            assert fuv.is_1d_array(ones) is True
            zeros = np.zeros(shape=shape, dtype=dtype)
            assert fuv.is_1d_array(zeros) is True
            empty = np.empty(shape=shape, dtype=dtype)
            assert fuv.is_1d_array(empty) is True
    for shape in flat_shapes:
        for dtype in flat_struct:
            ones = np.ones(shape=shape, dtype=dtype)
            with pytest.warns(UserWarning) as warning:
                assert fuv.is_1d_array(ones) is False
            assert warning_message == str(warning[0].message)
            #
            zeros = np.zeros(shape=shape, dtype=dtype)
            with pytest.warns(UserWarning) as warning:
                assert fuv.is_1d_array(zeros) is False
            assert warning_message == str(warning[0].message)
            #
            empty = np.empty(shape=shape, dtype=dtype)
            with pytest.warns(UserWarning) as warning:
                assert fuv.is_1d_array(empty) is False
            assert warning_message == str(warning[0].message)
    for shape in flat_shapes:
        for dtype in not_flat_dtype:
            ones = np.ones(shape=shape, dtype=dtype)
            assert fuv.is_1d_array(ones) is False
            zeros = np.zeros(shape=shape, dtype=dtype)
            assert fuv.is_1d_array(zeros) is False
            empty = np.empty(shape=shape, dtype=dtype)
            assert fuv.is_1d_array(empty) is False
    for shape in not_flat_shapes:
        for dtype in not_flat_dtype + flat_dtype + flat_struct:
            ones = np.ones(shape=shape, dtype=dtype)
            assert fuv.is_1d_array(ones) is False
            zeros = np.zeros(shape=shape, dtype=dtype)
            assert fuv.is_1d_array(zeros) is False
            empty = np.empty(shape=shape, dtype=dtype)
            assert fuv.is_1d_array(empty) is False


def test_is_structured_array():
    """
    Tests :func:`fatf.utils.validation.is_structured_array` function.
    """
    type_error = 'The input should be a numpy array-like.'
    # Test any object
    with pytest.raises(TypeError) as exin:
        fuv.is_structured_array(None)
    assert str(exin.value) == type_error

    assert fuv.is_structured_array(NUMERICAL_NP_ARRAY) is False
    assert fuv.is_structured_array(NOT_NUMERICAL_NP_ARRAY) is False
    assert fuv.is_structured_array(WIDE_NP_ARRAY) is False
    assert fuv.is_structured_array(NUMERICAL_STRUCTURED_ARRAY) is True
    assert fuv.is_structured_array(NOT_NUMERICAL_STRUCTURED_ARRAY) is True
    assert fuv.is_structured_array(WIDE_STRUCTURED_ARRAY) is True

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
            assert fuv.is_structured_array(ones) is False
            zeros = np.zeros(shape=shape, dtype=dtype)
            assert fuv.is_structured_array(zeros) is False
            empty = np.empty(shape=shape, dtype=dtype)
            assert fuv.is_structured_array(empty) is False
        for dtype in struct_dtype:
            ones = np.ones(shape=shape, dtype=dtype)
            assert fuv.is_structured_array(ones) is True
            zeros = np.zeros(shape=shape, dtype=dtype)
            assert fuv.is_structured_array(zeros) is True
            empty = np.empty(shape=shape, dtype=dtype)
            assert fuv.is_structured_array(empty) is True


def test_indices_by_type():
    """
    Tests :func:`fatf.utils.validation.indices_by_type` function.
    """
    # pylint: disable=too-many-locals,too-many-statements
    # Test any object and shape
    type_error = 'The input should be a numpy array-like.'
    incorrect_shape_error = 'The input array should be 2-dimensional.'
    with pytest.raises(TypeError) as exin:
        fuv.indices_by_type(None)
    assert str(exin.value) == type_error
    with pytest.raises(fatf.exceptions.IncorrectShapeError) as exin:
        fuv.indices_by_type(np.empty((0, )))
    assert str(exin.value) == incorrect_shape_error

    # Empty array
    i_n, i_c = fuv.indices_by_type(np.empty((22, 0)))
    assert np.array_equal([], i_n)
    assert np.array_equal([], i_c)

    # All numerical array
    array_all_numerical = np.ones((22, 4))
    array_all_numerical_indices_numerical = np.array([0, 1, 2, 3])
    array_all_numerical_indices_categorical = np.array([], dtype=int)
    i_n, i_c = fuv.indices_by_type(array_all_numerical)
    assert np.array_equal(array_all_numerical_indices_numerical, i_n)
    assert np.array_equal(array_all_numerical_indices_categorical, i_c)

    # All categorical -- single type -- array
    array_all_categorical = np.ones((22, 4), dtype='U4')
    array_all_categorical_indices_numerical = np.array([])
    array_all_categorical_indices_categorical = np.array([0, 1, 2, 3])
    i_n, i_c = fuv.indices_by_type(array_all_categorical)
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
    i_n, i_c = fuv.indices_by_type(array_mixture_1)
    assert np.array_equal(array_mixture_1_indices_numerical, i_n)
    assert np.array_equal(array_mixture_1_indices_categorical, i_c)

    array_mixture_2 = np.ones((22, ), dtype=[('a', 'U4'),
                                             ('b', 'f'),
                                             ('c', 'U4'),
                                             ('d', int)])  # yapf: disable
    array_mixture_2_indices_numerical = np.array(['b', 'd'], dtype='U1')
    array_mixture_2_indices_categorical = np.array(['a', 'c'], dtype='U1')
    i_n, i_c = fuv.indices_by_type(array_mixture_2)
    assert np.array_equal(array_mixture_2_indices_numerical, i_n)
    assert np.array_equal(array_mixture_2_indices_categorical, i_c)

    glob_indices_numerical = np.array([0, 1])
    glob_indices_categorical = np.array([])
    i_n, i_c = fuv.indices_by_type(NUMERICAL_NP_ARRAY)
    assert np.array_equal(glob_indices_numerical, i_n)
    assert np.array_equal(glob_indices_categorical, i_c)
    #
    glob_indices_numerical = np.array([])
    glob_indices_categorical = np.array([0, 1])
    i_n, i_c = fuv.indices_by_type(NOT_NUMERICAL_NP_ARRAY)
    assert np.array_equal(glob_indices_numerical, i_n)
    assert np.array_equal(glob_indices_categorical, i_c)
    #
    glob_indices_numerical = np.array([0, 1, 2])
    glob_indices_categorical = np.array([])
    i_n, i_c = fuv.indices_by_type(WIDE_NP_ARRAY)
    assert np.array_equal(glob_indices_numerical, i_n)
    assert np.array_equal(glob_indices_categorical, i_c)
    #
    glob_indices_numerical = np.array(['numbers', 'complex'])
    glob_indices_categorical = np.array([])
    i_n, i_c = fuv.indices_by_type(NUMERICAL_STRUCTURED_ARRAY)
    assert np.array_equal(glob_indices_numerical, i_n)
    assert np.array_equal(glob_indices_categorical, i_c)
    #
    glob_indices_numerical = np.array(['numerical'])
    glob_indices_categorical = np.array(['categorical'])
    i_n, i_c = fuv.indices_by_type(NOT_NUMERICAL_STRUCTURED_ARRAY)
    assert np.array_equal(glob_indices_numerical, i_n)
    assert np.array_equal(glob_indices_categorical, i_c)
    #
    glob_indices_numerical = np.array(['numbers', 'complex', 'anybody'])
    glob_indices_categorical = np.array([])
    i_n, i_c = fuv.indices_by_type(WIDE_STRUCTURED_ARRAY)
    assert np.array_equal(glob_indices_numerical, i_n)
    assert np.array_equal(glob_indices_categorical, i_c)


def test_get_invalid_indices():
    """
    Tests :func:`fatf.utils.validation.get_invalid_indices` function.
    """
    type_error = 'Input arrays should be numpy array-like objects.'
    incorrect_shape_array = 'The input array should be 2-dimensional.'
    incorrect_shape_indices = 'The indices array should be 1-dimensional.'
    with pytest.raises(TypeError) as exin:
        fuv.get_invalid_indices(None, np.ones((4, )))
    assert str(exin.value) == type_error
    with pytest.raises(TypeError) as exin:
        fuv.get_invalid_indices(None, np.ones((4, 4)))
    assert str(exin.value) == type_error
    with pytest.raises(TypeError) as exin:
        fuv.get_invalid_indices(np.ones((4, )), None)
    assert str(exin.value) == type_error
    with pytest.raises(TypeError) as exin:
        fuv.get_invalid_indices(None, np.ones((4, 4)))
    assert str(exin.value) == type_error
    # Incorrect shape array
    with pytest.raises(fatf.exceptions.IncorrectShapeError) as exin:
        fuv.get_invalid_indices(np.ones((5, )), np.ones((4, 4)))
    assert str(exin.value) == incorrect_shape_array
    with pytest.raises(fatf.exceptions.IncorrectShapeError) as exin:
        fuv.get_invalid_indices(np.ones((5, )), np.ones((4, )))
    assert str(exin.value) == incorrect_shape_array
    with pytest.raises(fatf.exceptions.IncorrectShapeError) as exin:
        fuv.get_invalid_indices(np.ones((5, 3)), np.ones((4, 4)))
    assert str(exin.value) == incorrect_shape_indices

    gind = fuv.get_invalid_indices(NUMERICAL_NP_ARRAY, np.array([0, 2]))
    assert np.array_equal(gind, np.array([2]))
    gind = fuv.get_invalid_indices(NUMERICAL_NP_ARRAY, np.array(['a', 1]))
    assert np.array_equal(gind, np.array(['1', 'a']))
    gind = fuv.get_invalid_indices(NUMERICAL_NP_ARRAY, np.array([1, 0]))
    assert np.array_equal(gind, np.array([]))
    assert np.array_equal(gind, np.empty((0, )))
    #
    gind = fuv.get_invalid_indices(NOT_NUMERICAL_NP_ARRAY, np.array([0, 2]))
    assert np.array_equal(gind, np.array([2]))
    gind = fuv.get_invalid_indices(NOT_NUMERICAL_NP_ARRAY, np.array(['a', 1]))
    assert np.array_equal(gind, np.array(['1', 'a']))
    #
    gind = fuv.get_invalid_indices(NUMERICAL_STRUCTURED_ARRAY,
                                   np.array([0, 'numbers']))
    assert np.array_equal(gind, np.array(['0']))
    gind = fuv.get_invalid_indices(NUMERICAL_STRUCTURED_ARRAY, np.array([0]))
    assert np.array_equal(gind, np.array([0]))
    gind = fuv.get_invalid_indices(NUMERICAL_STRUCTURED_ARRAY,
                                   np.array(['complex', 'numbers']))
    assert np.array_equal(gind, np.array([]))
    #
    gind = fuv.get_invalid_indices(WIDE_STRUCTURED_ARRAY,
                                   np.array(['complex', 'numbers']))
    assert np.array_equal(gind, np.array([]))


def test_are_indices_valid():
    """
    Tests :func:`fatf.utils.validation.are_indices_valid` function.
    """
    type_error = 'Input arrays should be numpy array-like objects.'
    incorrect_shape_array = 'The input array should be 2-dimensional.'
    incorrect_shape_indices = 'The indices array should be 1-dimensional.'
    with pytest.raises(TypeError) as exin:
        fuv.are_indices_valid(None, np.ones((4, )))
    assert str(exin.value) == type_error
    with pytest.raises(TypeError) as exin:
        fuv.are_indices_valid(None, np.ones((4, 4)))
    assert str(exin.value) == type_error
    with pytest.raises(TypeError) as exin:
        fuv.are_indices_valid(np.ones((4, )), None)
    assert str(exin.value) == type_error
    with pytest.raises(TypeError) as exin:
        fuv.are_indices_valid(None, np.ones((4, 4)))
    assert str(exin.value) == type_error
    # Incorrect shape array
    with pytest.raises(fatf.exceptions.IncorrectShapeError) as exin:
        fuv.are_indices_valid(np.ones((5, )), np.ones((4, 4)))
    assert str(exin.value) == incorrect_shape_array
    with pytest.raises(fatf.exceptions.IncorrectShapeError) as exin:
        fuv.are_indices_valid(np.ones((5, )), np.ones((4, )))
    assert str(exin.value) == incorrect_shape_array
    with pytest.raises(fatf.exceptions.IncorrectShapeError) as exin:
        fuv.are_indices_valid(np.ones((5, 3)), np.ones((4, 4)))
    assert str(exin.value) == incorrect_shape_indices

    assert not fuv.are_indices_valid(NUMERICAL_NP_ARRAY, np.array([0, 2]))
    assert not fuv.are_indices_valid(NUMERICAL_NP_ARRAY, np.array(['a', 1]))
    assert fuv.are_indices_valid(NUMERICAL_NP_ARRAY, np.array([1, 0]))
    #
    assert not fuv.are_indices_valid(NOT_NUMERICAL_NP_ARRAY, np.array([0, 2]))
    assert not fuv.are_indices_valid(NOT_NUMERICAL_NP_ARRAY,
                                     np.array(['a', 1]))  # yapf: disable
    assert fuv.are_indices_valid(NOT_NUMERICAL_NP_ARRAY, np.array([0, 1]))
    #
    assert not fuv.are_indices_valid(NUMERICAL_STRUCTURED_ARRAY,
                                     np.array([0, 'numbers']))
    assert not fuv.are_indices_valid(NUMERICAL_STRUCTURED_ARRAY, np.array([0]))
    assert fuv.are_indices_valid(NUMERICAL_STRUCTURED_ARRAY,
                                 np.array(['complex', 'numbers']))
    #
    assert fuv.are_indices_valid(WIDE_STRUCTURED_ARRAY,
                                 np.array(['complex', 'numbers']))


def test_check_model_functionality():
    """
    Tests :func:`fatf.utils.validation.check_model_functionality` function.
    """  # yapf: disable
    # yapf: disable
    # pylint: disable=unused-variable,useless-object-inheritance
    # pylint: disable=too-few-public-methods,missing-docstring
    # pylint: disable=multiple-statements,too-many-locals,too-many-statements
    class ClassPlain(object): pass
    class_plain = ClassPlain()
    class ClassObject(object): pass
    class_object = ClassObject()
    class ClassInit0(object):
        def __init__(self): pass
    class_init_0 = ClassInit0()
    class ClassInit1(object):
        def __init__(self, one): pass
    class_init_1 = ClassInit1(1)
    class ClassFit0(object):
        def fit(self): pass
    class_fit_0 = ClassFit0()
    class ClassFit1(object):
        def fit(self, one): pass
    class_fit_1 = ClassFit1()
    class ClassFit11(object):
        def fit(self, one, two=2): pass
    class_fit_11 = ClassFit11()
    class ClassFit2(object):
        def fit(self, one, two): pass
    class_fit_2 = ClassFit2()
    class ClassFit21(object):
        def fit(self, one, two, three=3): pass
    class_fit_21 = ClassFit21()
    class ClassFit3(object):
        def fit(self, one, two, three): pass
    class_fit_3 = ClassFit3()
    class ClassPredict0(object):
        def predict(self): pass
    class_predict_0 = ClassPredict0()
    class ClassPredict1(object):
        def predict(self, one): pass
    class_predict_1 = ClassPredict1()
    class ClassPredict2(object):
        def predict(self, one, two): pass
    class_predict_2 = ClassPredict2()
    class ClassPredictProba0(object):
        def predict_proba(self): pass
    class_predict_proba_0 = ClassPredictProba0()
    class ClassPredictProba1(object):
        def predict_proba(self, one): pass
    class_predict_proba_1 = ClassPredictProba1()
    class ClassPredictProba2(object):
        def predict_proba(self, one, two): pass
    class_predict_proba_2 = ClassPredictProba2()

    class ClassFit11Predict1(ClassFit11, ClassPredict1): pass
    class_fit_11_predict_1 = ClassFit11Predict1()
    class ClassFit21Predict1(ClassFit21, ClassPredict1): pass
    class_fit_21_predict_1 = ClassFit21Predict1()

    class ClassFit1Predict2(ClassFit1, ClassPredict2): pass
    class_fit_1_predict_2 = ClassFit1Predict2()
    class ClassFit3Predict0(ClassFit3, ClassPredict0): pass
    class_fit_3_predict_0 = ClassFit3Predict0()
    class ClassFit3Predict1PredictProba0(ClassFit3, ClassPredict1,
                                         ClassPredictProba0):
        pass
    class_fit_3_predict_1_predict_proba_0 = ClassFit3Predict1PredictProba0()

    class ClassFit2Predict1(ClassFit2, ClassPredict1): pass
    class_fit_2_predict_1 = ClassFit2Predict1()
    class ClassFit2Predict1PredictProba1(ClassFit2, ClassPredict1,
                                         ClassPredictProba1):
        pass
    class_fit_2_predict_1_predict_proba_1 = ClassFit2Predict1PredictProba1()
    class ClassFit2Predict1PredictProba0(ClassFit2, ClassPredict1,
                                         ClassPredictProba0):
        pass
    class_fit_2_predict_1_predict_proba_0 = ClassFit2Predict1PredictProba0()
    # yapf: enable

    # Test not suppressed -- warning
    with pytest.warns(UserWarning) as warning:
        assert fuv.check_model_functionality(class_plain, True, False) is False
    w_message = str(warning[0].message)
    assert ('missing \'fit\'' in w_message
            and 'missing \'predict\'' in w_message
            and 'missing \'predict_proba\'' in w_message)

    # Test suppressed -- warning
    assert fuv.check_model_functionality(class_plain, True, True) is False

    # Test optional arguments
    assert fuv.check_model_functionality(
        class_fit_11_predict_1, suppress_warning=True) is False
    assert fuv.check_model_functionality(class_fit_21_predict_1) is True

    # Too few method parameters
    with pytest.warns(UserWarning) as warning:
        assert fuv.check_model_functionality(
            class_fit_1_predict_2, suppress_warning=False) is False
    w_message = str(warning[0].message)
    m_message_1 = ('The \'fit\' method of the class has incorrect number '
                   '(1) of the required parameters. It needs to have exactly'
                   ' 2 required parameters. Try using optional '
                   'parameters if you require more functionality.')
    m_message_2 = ('The \'predict\' method of the class has incorrect number '
                   '(2) of the required parameters. It needs to have exactly'
                   ' 1 required parameters. Try using optional '
                   'parameters if you require more functionality.')
    assert m_message_1 in w_message and m_message_2 in w_message

    with pytest.warns(UserWarning) as warning:
        assert fuv.check_model_functionality(
            class_fit_3_predict_0, suppress_warning=False) is False
    w_message = str(warning[0].message)
    m_message_1 = ('The \'fit\' method of the class has incorrect number '
                   '(3) of the required parameters. It needs to have exactly'
                   ' 2 required parameters. Try using optional '
                   'parameters if you require more functionality.')
    m_message_2 = ('The \'predict\' method of the class has incorrect number '
                   '(0) of the required parameters. It needs to have exactly'
                   ' 1 required parameters. Try using optional '
                   'parameters if you require more functionality.')
    assert m_message_1 in w_message and m_message_2 in w_message

    with pytest.warns(UserWarning) as warning:
        assert fuv.check_model_functionality(class_fit_3_predict_0, True,
                                             False) is False
    w_message = str(warning[0].message)
    m_message_1 = ('The \'fit\' method of the class has incorrect number '
                   '(3) of the required parameters. It needs to have exactly'
                   ' 2 required parameters. Try using optional '
                   'parameters if you require more functionality.')
    m_message_2 = ('The \'predict\' method of the class has incorrect number '
                   '(0) of the required parameters. It needs to have exactly'
                   ' 1 required parameters. Try using optional '
                   'parameters if you require more functionality.')
    assert (m_message_1 in w_message and m_message_2 in w_message
            and 'missing \'predict_proba\'' in w_message)

    assert fuv.check_model_functionality(
        class_fit_2_predict_1_predict_proba_0) is True
    assert fuv.check_model_functionality(
        class_fit_2_predict_1_predict_proba_0, True,
        suppress_warning=True) is False
    assert fuv.check_model_functionality(
        class_fit_3_predict_1_predict_proba_0, suppress_warning=True) is False
    assert fuv.check_model_functionality(
        class_fit_3_predict_1_predict_proba_0, True,
        suppress_warning=True) is False

    # Test predict_proba
    assert fuv.check_model_functionality(class_fit_2_predict_1) is True
    assert fuv.check_model_functionality(
        class_fit_2_predict_1, True, suppress_warning=True) is False
    assert fuv.check_model_functionality(class_fit_2_predict_1_predict_proba_1,
                                         False) is True
    assert fuv.check_model_functionality(class_fit_2_predict_1_predict_proba_1,
                                         True) is True
