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
    (True, 'a'),
    (1, 'b'),
    (-1, 'c'),
    (1.0, 'd'),
    (1 + 1j, 'e'),
    (False, 'f')], dtype=[('numerical', 'c8'),
                          ('categorical', 'U1')])  # yapf: disable

NP_VER = [int(i) for i in np.version.version.split('.')]
NP_VER_TYPEERROR_MSG = 'a bytes-like object is required, not \'int\''


def test_is_numerical_dtype():
    """
    Tests :func:`fatf.utils.validation.is_numerical_dtype` function.
    """
    # Test any object
    object_1 = None
    with pytest.raises(TypeError):
        fuv.is_numerical_dtype(object_1)

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

    # Test complex numerical arrays
    assert fuv.is_numerical_dtype(NUMERICAL_NP_ARRAY.dtype) is True

    # Test complex not numerical arrays
    assert fuv.is_numerical_dtype(NOT_NUMERICAL_NP_ARRAY.dtype) is False

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
    # Test any object
    object_1 = None
    with pytest.raises(TypeError):
        fuv.is_numerical_array(object_1)

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

    # Test complex numerical arrays
    assert fuv.is_numerical_array(NUMERICAL_NP_ARRAY) is True

    # Test complex not numerical arrays
    assert fuv.is_numerical_array(NOT_NUMERICAL_NP_ARRAY) is False

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
                else:
                    assert fuv.is_numerical_array(
                        np.empty((1, ), dtype=dtype)) is False
                    assert fuv.is_numerical_array(
                        np.zeros((1, ), dtype=dtype)) is False
                    if (dtype is np.void and NP_VER[0] < 2
                            and NP_VER[1] < 14):  # pragma: no cover
                        with pytest.raises(TypeError) as exin:
                            fuv.is_numerical_array(np.ones((1, ), dtype=dtype))
                        assert str(exin.value) == NP_VER_TYPEERROR_MSG
                    else:
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
    # pylint: disable=too-many-branches
    # Test any object
    with pytest.raises(TypeError):
        fuv.is_2d_array(None)

    # Test simple numerical and not numerical arrays
    for i in NUMERICAL_KINDS + NOT_NUMERICAL_KINDS:
        for j in [[[i] * 2] * 2]:
            assert fuv.is_2d_array(np.array(j)) is True
        for j in [i, [i], [i] * 2, [[[i] * 2] * 2] * 2]:
            assert fuv.is_2d_array(np.array(j)) is False

    # Test complex numerical and not numerical arrays
    assert fuv.is_2d_array(NUMERICAL_NP_ARRAY) is True
    assert fuv.is_2d_array(NOT_NUMERICAL_NP_ARRAY) is True

    square_shapes = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 1), (2, 2)]
    not_square_shapes = [(0, ), (1, ), (2, ), (0, 0, 0), (1, 0, 0), (0, 1, 0),
                         (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1),
                         (2, 2, 2), (2, 1, 1), (2, 2, 1)]
    for _, dtypes in np.sctypes.items():
        for dtype in dtypes:
            for shape in square_shapes:
                if (dtype is np.void and 0 not in shape and NP_VER[0] < 2
                        and NP_VER[1] < 14):  # pragma: no cover
                    with pytest.raises(TypeError) as exin:
                        np.ones(shape=shape, dtype=dtype)
                    assert str(exin.value) == NP_VER_TYPEERROR_MSG
                else:
                    ones = np.ones(shape=shape, dtype=dtype)
                    assert fuv.is_2d_array(ones) is True
                zeros = np.zeros(shape=shape, dtype=dtype)
                assert fuv.is_2d_array(zeros) is True
                empty = np.empty(shape=shape, dtype=dtype)
                assert fuv.is_2d_array(empty) is True
            for shape in not_square_shapes:
                if (dtype is np.void and 0 not in shape and NP_VER[0] < 2
                        and NP_VER[1] < 14):  # pragma: no cover
                    with pytest.raises(TypeError) as exin:
                        np.ones(shape=shape, dtype=dtype)
                    assert str(exin.value) == NP_VER_TYPEERROR_MSG
                else:
                    ones = np.ones(shape=shape, dtype=dtype)
                    assert fuv.is_2d_array(ones) is False
                zeros = np.zeros(shape=shape, dtype=dtype)
                assert fuv.is_2d_array(zeros) is False
                empty = np.empty(shape=shape, dtype=dtype)
                assert fuv.is_2d_array(empty) is False

    complex_dtype = NOT_NUMERICAL_NP_ARRAY.dtype
    complex_square_shapes = [(0, ), (1, ), (2, )]
    complex_not_square_shapes = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 1),
                                 (2, 2), (0, 0, 0), (1, 0, 0), (0, 1, 0),
                                 (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1),
                                 (1, 1, 1), (2, 2, 2), (2, 1, 1),
                                 (2, 2, 1)]  # yapf: disable
    # Structured arrays flat with multi-demnsional tuples
    for shape in complex_square_shapes:
        ones = np.ones(shape=shape, dtype=complex_dtype)
        assert fuv.is_2d_array(ones) is True
        zeros = np.zeros(shape=shape, dtype=complex_dtype)
        assert fuv.is_2d_array(zeros) is True
        empty = np.empty(shape=shape, dtype=complex_dtype)
        assert fuv.is_2d_array(empty) is True
    for shape in complex_not_square_shapes:
        ones = np.ones(shape=shape, dtype=complex_dtype)
        assert fuv.is_2d_array(ones) is False
        zeros = np.zeros(shape=shape, dtype=complex_dtype)
        assert fuv.is_2d_array(zeros) is False
        empty = np.empty(shape=shape, dtype=complex_dtype)
        assert fuv.is_2d_array(empty) is False


def test_is_1d_array():
    """
    Tests :func:`fatf.utils.validation.is_1d_array` function.
    """
    # pylint: disable=too-many-branches
    # Test any object
    with pytest.raises(TypeError):
        fuv.is_1d_array(None)

    complex_type_msg = ('Structured numpy arrays cannot be 1-dimensional. Please '
                   'use a classic numpy array with a specified type.')
    # Test simple numerical and not numerical arrays
    for i in NUMERICAL_KINDS + NOT_NUMERICAL_KINDS:
        for j in [[i], [i] * 2]:
            assert fuv.is_1d_array(np.array(j)) is True
        for j in [i, [[i] * 2] * 2, [[[i] * 2] * 2] * 2]:
            assert fuv.is_1d_array(np.array(j)) is False

    # Test complex numerical and not numerical arrays
    assert fuv.is_1d_array(NUMERICAL_NP_ARRAY) is False
    with pytest.warns(UserWarning) as warning:
        assert fuv.is_1d_array(NOT_NUMERICAL_NP_ARRAY) is False
    assert complex_type_msg == str(warning[0].message)
    flat_shapes = [(0, ), (1, ), (2, )]
    not_flat_shapes = [(0, 0, 0), (1, 0, 0), (0, 1, 0),
                         (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1),
                         (2, 2, 2), (2, 1, 1), (2, 2, 1), (0, 0), (0, 1), 
                         (1, 0), (1, 1), (2, 1), (2, 2)]
    for _, dtypes in np.sctypes.items():
        for dtype in dtypes:
            for shape in flat_shapes:
                if (dtype is np.void and 0 not in shape and NP_VER[0] < 2
                        and NP_VER[1] < 14):  # pragma: no cover
                    with pytest.raises(TypeError) as exin:
                        np.ones(shape=shape, dtype=dtype)
                    assert str(exin.value) == NP_VER_TYPEERROR_MSG
                else:
                    ones = np.ones(shape=shape, dtype=dtype)
                    assert fuv.is_1d_array(ones) is True
                zeros = np.zeros(shape=shape, dtype=dtype)
                assert fuv.is_1d_array(zeros) is True
                empty = np.empty(shape=shape, dtype=dtype)
                assert fuv.is_1d_array(empty) is True
            for shape in not_flat_shapes:
                if (dtype is np.void and 0 not in shape and NP_VER[0] < 2
                        and NP_VER[1] < 14):  # pragma: no cover
                    with pytest.raises(TypeError) as exin:
                        np.ones(shape=shape, dtype=dtype)
                    assert str(exin.value) == NP_VER_TYPEERROR_MSG
                else:
                    ones = np.ones(shape=shape, dtype=dtype)
                    assert fuv.is_1d_array(ones) is False
                zeros = np.zeros(shape=shape, dtype=dtype)
                assert fuv.is_1d_array(zeros) is False
                empty = np.empty(shape=shape, dtype=dtype)
                assert fuv.is_1d_array(empty) is False

    complex_dtype = NOT_NUMERICAL_NP_ARRAY.dtype
    complex_not_flat_shapes = [(0, ), (1, ), (2, ), (0, 0), (0, 1), (1, 0), (1, 1), (2, 1),
                                 (2, 2), (0, 0, 0), (1, 0, 0), (0, 1, 0),
                                 (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1),
                                 (1, 1, 1), (2, 2, 2), (2, 1, 1),
                                 (2, 2, 1)]  # yapf: disable
    for shape in complex_not_flat_shapes:
        ones = np.ones(shape=shape, dtype=complex_dtype)
        with pytest.warns(UserWarning) as warning:
            assert fuv.is_1d_array(ones) is False
        assert complex_type_msg == str(warning[0].message)
        zeros = np.zeros(shape=shape, dtype=complex_dtype)
        with pytest.warns(UserWarning) as warning:
            assert fuv.is_1d_array(zeros) is False
        assert complex_type_msg == str(warning[0].message)
        empty = np.empty(shape=shape, dtype=complex_dtype)
        with pytest.warns(UserWarning) as warning:
            assert fuv.is_1d_array(empty) is False
        assert complex_type_msg == str(warning[0].message)


def test_is_structured():
    """
    Tests :func:`fatf.utils.validation.is_structured` function.
    """
    array_all_numerical_structured = np.ones((22,),
                                             dtype=[('a', 'f'),
                                                    ('b', 'f'),
                                                    ('c', int),
                                                    ('d', int)])
    array_all_numerical = np.ones((22,4))
    array_mixture = np.ones((22,),
                             dtype=[('a', 'U4'),
                                    ('b', 'f'),
                                    ('c', 'U4'),
                                    ('d', int)])
    array_all_categorical = np.ones((22,4), dtype='U4')
    results = [True, False, True, False]
    arrays = [array_all_numerical_structured, array_all_numerical, array_mixture,
              array_all_categorical]
    for array, res in zip(arrays, results):
        assert(fuv.is_structured(array) == res)


def test_indices_by_type():
    """
    Tests :func:`fatf.utils.validation.indices_by_type` function.
    """
    # Test any object
    with pytest.raises(TypeError):
        fuv.indices_by_type(None)
    with pytest.raises(fatf.exceptions.IncorrectShapeError):
        fuv.indices_by_type(np.empty((0, )))

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


def test_check_indices():
    inputs = [np.array([0, 1, 2]), 
              np.array(['a', 'b', 'c']),
              np.array(['a', 'b', 'z']),
              np.array([0, 2, 6]),
              np.array([('a', 0)], dtype=[('a', 'U4'), ('b', 'int')]),
              np.array([-1, 0, 1]),
              np.array([[1, 0], [0, 2]])]
    non_structured_outputs = [np.array([], dtype=np.int64),
                              np.array(['a', 'b', 'c']),
                              np.array(['a', 'b', 'z']),
                              np.array([6]),
                              np.array([('a', 0)], dtype=[('a', 'U4'), ('b', 'int')]),
                              np.array([-1]),
                              np.array([[1, 0], [0, 2]])]
    structured_outputs = [np.array([0, 1, 2], dtype=np.int64),
                          np.array([], dtype='U1'),
                          np.array(['z']),
                          np.array([0, 2, 6]),
                          np.array([('a', 0)], dtype=[('a', 'U4'), ('b', 'int')]),
                          np.array([-1, 0, 1]),
                          np.array([[1, 0], [0, 2]])]
    non_structured_valid_outputs = [True, False, False, False, False, False, False]
    structured_valid_outputs = [False, True, False, False, False, False, False]
    array_all_numerical = np.ones((22,4))
    for input_array, output, valid in zip(inputs, non_structured_outputs, 
                                          non_structured_valid_outputs):
        assert(np.array_equal(fuv.check_indices(array_all_numerical, input_array), 
                              output))
        assert(fuv.check_valid_indices(array_all_numerical, input_array) == valid)

    array_all_categorical = np.ones((22,4), dtype='U4')
    for input_array, output, valid in zip(inputs, non_structured_outputs,
                                          non_structured_valid_outputs):
        assert(np.array_equal(fuv.check_indices(array_all_categorical, input_array), 
                              output))
        assert(fuv.check_valid_indices(array_all_categorical, input_array) == valid)

    array_mixture = np.ones((22,),
                              dtype=[('a', 'U4'),
                                     ('b', 'f'),
                                     ('c', 'U4'),
                                     ('d', int)])
    for input_array, output, valid in zip(inputs, structured_outputs, 
                                          structured_valid_outputs):
        assert(np.array_equal(fuv.check_indices(array_mixture, input_array), output))
        assert(fuv.check_valid_indices(array_mixture, input_array) == valid)

    array_all_numerical_structured = np.ones((22,),
                                            dtype=[('a', 'f'),
                                                   ('b', 'f'),
                                                   ('c', int),
                                                   ('d', int)])
    for input_array, output, valid in zip(inputs, structured_outputs,
                                          structured_valid_outputs):
        assert(np.array_equal(fuv.check_indices(array_all_numerical_structured, input_array), 
                              output))
        assert(fuv.check_valid_indices(array_all_numerical_structured, input_array)
               == valid)


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
