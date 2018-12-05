import numpy as np
import pytest

import fatf.exceptions
import fatf.utils.validation as fuv

NUMERICAL_KINDS = [True, 1, -1, 1.0, 1+1j]
NOT_NUMERICAL_KINDS = [object(), 'string', u'unicode', None]

NUMERICAL_NP_ARRAY = np.array([
        [True, 1],
        [-1, 1.0],
        [1+1j, False]
    ])
NOT_NUMERICAL_NP_ARRAY = np.array([
        (True, 'a'),
        (1, 'b'),
        (-1, 'c'),
        (1.0, 'd'),
        (1+1j, 'e'),
        (False, 'f')
    ], dtype=[('numerical', 'c8'), ('categorical', 'U1')])

def test_is_numerical_dtype():
    # Test any object
    object_1 = None
    with pytest.raises(fatf.exceptions.CustomValueError):
        fuv.is_numerical_dtype(object_1)

    # Test simple numerical arrays
    for i in NUMERICAL_KINDS:
        for j in [i, [i], [i]*2, [[i] *2 ] * 2]:
            array = np.array(j)
            array_dtype = array.dtype
            assert fuv.is_numerical_dtype(array_dtype) == True

    # Test simple not numerical arrays
    for i in NOT_NUMERICAL_KINDS:
        for j in [i, [i], [i]*2, [[i] *2 ] * 2]:
            array = np.array(j)
            array_dtype = array.dtype
            assert fuv.is_numerical_dtype(array_dtype) == False

    # Test complex numerical arrays
    assert fuv.is_numerical_dtype(NUMERICAL_NP_ARRAY.dtype) == True

    # Test complex not numerical arrays
    assert fuv.is_numerical_dtype(NOT_NUMERICAL_NP_ARRAY.dtype) == False

    # Test numpy types
    for kind, dtypes in np.sctypes.items():
        if kind == 'others':
            for dtype in dtypes:
                if dtype is bool:
                    assert fuv.is_numerical_dtype(np.dtype(dtype)) == True
                else:
                    assert fuv.is_numerical_dtype(np.dtype(dtype)) == False
        else:
            for dtype in dtypes:
                assert fuv.is_numerical_dtype(np.dtype(dtype)) == True

def test_is_numerical_array():
    # Test any object
    object_1 = None
    with pytest.raises(fatf.exceptions.CustomValueError):
        fuv.is_numerical_array(object_1)

    # Test simple numerical arrays
    for i in NUMERICAL_KINDS:
        for j in [i, [i], [i]*2, [[i] *2 ] * 2]:
            array = np.array(j)
            assert fuv.is_numerical_array(array) == True

    # Test simple not numerical arrays
    for i in NOT_NUMERICAL_KINDS:
        for j in [i, [i], [i]*2, [[i] *2 ] * 2]:
            array = np.array(j)
            assert fuv.is_numerical_array(array) == False

    # Test complex numerical arrays
    assert fuv.is_numerical_array(NUMERICAL_NP_ARRAY) == True

    # Test complex not numerical arrays
    assert fuv.is_numerical_array(NOT_NUMERICAL_NP_ARRAY) == False

    # Test numpy types
    for kind, dtypes in np.sctypes.items():
        if kind == 'others':
            for dtype in dtypes:
                if dtype is bool:
                    assert fuv.is_numerical_array(np.empty((1,), dtype=dtype)) \
                            == True
                    assert fuv.is_numerical_array(np.ones((1,), dtype=dtype)) \
                            == True
                    assert fuv.is_numerical_array(np.zeros((1,), dtype=dtype)) \
                            == True
                else:
                    assert fuv.is_numerical_array(np.empty((1,), dtype=dtype)) \
                            == False
                    assert fuv.is_numerical_array(np.ones((1,), dtype=dtype)) \
                            == False
                    assert fuv.is_numerical_array(np.zeros((1,), dtype=dtype)) \
                            == False
        else:
            for dtype in dtypes:
                assert fuv.is_numerical_array(np.empty((1,), dtype=dtype)) \
                        == True
                assert fuv.is_numerical_array(np.ones((1,), dtype=dtype)) \
                        == True
                assert fuv.is_numerical_array(np.zeros((1,), dtype=dtype)) \
                        == True

def test_is_2d_array():
    # Test any object
    object_1 = None
    with pytest.raises(fatf.exceptions.CustomValueError):
        fuv.is_2d_array(object_1)

    # Test simple numerical and not numerical arrays
    for i in NUMERICAL_KINDS + NOT_NUMERICAL_KINDS:
        for j in [[[i] *2 ] * 2]:
            array = np.array(j)
            assert fuv.is_2d_array(array) == True
        for j in [i, [i], [i]*2, [[[i] *2 ] * 2] * 2]:
            array = np.array(j)
            assert fuv.is_2d_array(array) == False

    # Test complex numerical and not numerical arrays
    assert fuv.is_2d_array(NUMERICAL_NP_ARRAY) == True
    assert fuv.is_2d_array(NOT_NUMERICAL_NP_ARRAY) == True

    square_shapes = [(0,0), (0,1), (1,0), (1,1), (2,1), (2,2)]
    not_square_shapes = [(0,), (1,), (2,), (0,0,0), (1,0,0), (0,1,0), (0,0,1),
                         (1,1,0), (0,1,1), (1,0,1), (1,1,1), (2,2,2), (2,1,1),
                         (2,2,1)]
    for _, dtypes in np.sctypes.items():
        for dtype in dtypes:
            for shape in square_shapes:
                ones = np.ones(shape=shape, dtype=dtype)
                assert fuv.is_2d_array(ones) == True
                zeros = np.zeros(shape=shape, dtype=dtype)
                assert fuv.is_2d_array(zeros) == True
                empty = np.empty(shape=shape, dtype=dtype)
                assert fuv.is_2d_array(empty) == True
            for shape in not_square_shapes:
                ones = np.ones(shape=shape, dtype=dtype)
                assert fuv.is_2d_array(ones) == False
                zeros = np.zeros(shape=shape, dtype=dtype)
                assert fuv.is_2d_array(zeros) == False
                empty = np.empty(shape=shape, dtype=dtype)
                assert fuv.is_2d_array(empty) == False

    complex_dtype = NOT_NUMERICAL_NP_ARRAY.dtype
    complex_square_shapes = [(0,), (1,), (2,)]
    complex_not_square_shapes = [(0,0), (0,1), (1,0), (1,1), (2,1), (2,2),
                                 (0,0,0), (1,0,0), (0,1,0), (0,0,1), (1,1,0),
                                 (0,1,1), (1,0,1), (1,1,1), (2,2,2), (2,1,1),
                                 (2,2,1)]
    for shape in complex_square_shapes:
        ones = np.ones(shape=shape, dtype=complex_dtype)
        assert fuv.is_2d_array(ones) == True
        zeros = np.zeros(shape=shape, dtype=complex_dtype)
        assert fuv.is_2d_array(zeros) == True
        empty = np.empty(shape=shape, dtype=complex_dtype)
        assert fuv.is_2d_array(empty) == True
    for shape in complex_not_square_shapes:
        ones = np.ones(shape=shape, dtype=complex_dtype)
        assert fuv.is_2d_array(ones) == False
        zeros = np.zeros(shape=shape, dtype=complex_dtype)
        assert fuv.is_2d_array(zeros) == False
        empty = np.empty(shape=shape, dtype=complex_dtype)
        assert fuv.is_2d_array(empty) == False

def test_check_array_type():
    # Test any object
    object_1 = None
    object_2 = np.empty((0,))
    with pytest.raises(fatf.exceptions.CustomValueError):
        fuv.check_array_type(object_1)
    with pytest.raises(fatf.exceptions.IncorrectShapeException):
        fuv.check_array_type(object_2)

    # Empty array
    array_empty = np.empty((22,0))
    i_n, i_c = fuv.check_array_type(array_empty)
    assert np.array_equal([], i_n)
    assert np.array_equal([], i_c)

    # All numerical array
    array_all_numerical = np.ones((22,4))
    array_all_numerical_indices_numerical = np.array([0,1,2,3])
    array_all_numerical_indices_categorical = np.array([], dtype=int)
    i_n, i_c = fuv.check_array_type(array_all_numerical)
    assert np.array_equal(array_all_numerical_indices_numerical, i_n)
    assert np.array_equal(array_all_numerical_indices_categorical, i_c)

    # All categorical -- single type -- array
    array_all_categorical = np.ones((22,4), dtype='U4')
    array_all_categorical_indices_numerical = np.array([])
    array_all_categorical_indices_categorical = np.array([0,1,2,3])
    i_n, i_c = fuv.check_array_type(array_all_categorical)
    assert np.array_equal(array_all_categorical_indices_numerical, i_n)
    assert np.array_equal(array_all_categorical_indices_categorical, i_c)

    # Mixture array
    array_mixture_1 = np.ones((22,),
                              dtype=[('a', 'U4'),
                                     ('b', 'U4'),
                                     ('c', 'U4'),
                                     ('d', 'U4')]
                             )
    array_mixture_1_indices_numerical = np.array([])
    array_mixture_1_indices_categorical = np.array(['a','b','c','d'],
                                                   dtype='U1')
    i_n, i_c = fuv.check_array_type(array_mixture_1)
    assert np.array_equal(array_mixture_1_indices_numerical, i_n)
    assert np.array_equal(array_mixture_1_indices_categorical, i_c)

    array_mixture_2 = np.ones((22,),
                              dtype=[('a', 'U4'),
                                     ('b', 'f'),
                                     ('c', 'U4'),
                                     ('d', int)]
                             )
    array_mixture_2_indices_numerical = np.array(['b','d'], dtype='U1')
    array_mixture_2_indices_categorical = np.array(['a','c'], dtype='U1')
    i_n, i_c = fuv.check_array_type(array_mixture_2)
    assert np.array_equal(array_mixture_2_indices_numerical, i_n)
    assert np.array_equal(array_mixture_2_indices_categorical, i_c)

def test_check_categorical_indices():
    array_all_numerical = np.ones((22,4))
    assert(fuv.check_indices(array_all_numerical, np.array([0, 1, 2, 3])) == True)
    assert(fuv.check_indices(array_all_numerical, np.array(['a', 'b'])) == False)
    assert(fuv.check_indices(array_all_numerical, np.array([0, 1, 2, 6])) == False)
    assert(fuv.check_indices(array_all_numerical, np.array(['a', 'b', 0, 1])) == False)
    assert(fuv.check_indices(array_all_numerical, np.array([-1, 0, 1, 2])) == False)
    assert(fuv.check_indices(array_all_numerical, np.array([[1, 0], [0, 2]])) == False)
    

    array_all_categorical = np.ones((22,4), dtype='U4')
    assert(fuv.check_indices(array_all_categorical, np.array([0, 1, 2, 3])) == True)
    assert(fuv.check_indices(array_all_categorical, np.array(['a', 'b'])) == False)
    assert(fuv.check_indices(array_all_categorical, np.array([0, 1, 2, 6])) == False)
    assert(fuv.check_indices(array_all_categorical, np.array(['a', 'b', 0, 1])) == False)
    assert(fuv.check_indices(array_all_categorical, np.array([-1, 0, 1, 2])) == False)
    assert(fuv.check_indices(array_all_numerical, np.array([[1, 0], [0, 2]])) == False)

    array_mixture = np.ones((22,),
                              dtype=[('a', 'U4'),
                                     ('b', 'f'),
                                     ('c', 'U4'),
                                     ('d', int)]
                             )
    assert(fuv.check_indices(array_mixture, np.array(['a', 'b', 'c', 'd'])) == True)
    assert(fuv.check_indices(array_mixture, np.array([0, 1, 2, 3])) == False)
    assert(fuv.check_indices(array_mixture, np.array(['a', 'b', 0, 1])) == False)
    assert(fuv.check_indices(array_mixture, np.array(['ads', 'f', 'a', 'b'])) == False)
    assert(fuv.check_indices(array_mixture, np.array([[1, 0], [0, 2]])) == False)
    
    array_all_numerical_structure = np.ones((22,),
                                            dtype=[('a', 'f'),
                                                   ('b', 'f'),
                                                   ('c', int),
                                                   ('d', int)]
                                           )
    assert(fuv.check_indices(array_all_numerical_structure, np.array(['a', 'b', 'c', 'd'])) == True)
    assert(fuv.check_indices(array_all_numerical_structure, np.array([0, 1, 2, 3])) == False)
    assert(fuv.check_indices(array_all_numerical_structure, np.array(['a', 'b', 0, 1])) == False)
    assert(fuv.check_indices(array_all_numerical_structure, np.array(['ads', 'f', 'a', 'b'])) == False)
    assert(fuv.check_indices(array_all_numerical_structure, np.array([[1, 0], [0, 2]])) == False)

def test_check_model_functionality():
    class ClassPlain: pass
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

    class ClassFit1Predict2(ClassFit1, ClassPredict2):
        pass
    class_fit_1_predict_2 = ClassFit1Predict2()
    class ClassFit3Predict0(ClassFit3, ClassPredict0):
        pass
    class_fit_3_predict_0 = ClassFit3Predict0()
    class ClassFit3Predict1PredictProba0(ClassFit3,
                                         ClassPredict1,
                                         ClassPredictProba0):
        pass
    class_fit_3_predict_1_predict_proba_0 = ClassFit3Predict1PredictProba0()

    class ClassFit2Predict1(ClassFit2, ClassPredict1):
        pass
    class_fit_2_predict_1 = ClassFit2Predict1()
    class ClassFit2Predict1PredictProba1(ClassFit2,
                                         ClassPredict1,
                                         ClassPredictProba1):
        pass
    class_fit_2_predict_1_predict_proba_1 = ClassFit2Predict1PredictProba1()
    class ClassFit2Predict1PredictProba0(ClassFit2,
                                         ClassPredict1,
                                         ClassPredictProba0):
        pass
    class_fit_2_predict_1_predict_proba_0 = ClassFit2Predict1PredictProba0()

    # Test verbose -- warning
    with pytest.warns(Warning) as w:
        assert fuv.check_model_functionality(class_plain, True, True) == False
    w_message = str(w[0].message)
    assert ('missing \'fit\'' in w_message and
            'missing \'predict\'' in w_message and
            'missing \'predict_proba\'' in w_message)

    # Test not verbose -- warning
    assert fuv.check_model_functionality(class_plain, True, False) == False
    assert fuv.check_model_functionality(class_plain, True) == False

    # Test optional arguments
    assert fuv.check_model_functionality(class_fit_11_predict_1) == False
    assert fuv.check_model_functionality(class_fit_21_predict_1) == True

    # Too few method parameters
    with pytest.warns(Warning) as w:
        assert fuv.check_model_functionality(class_fit_1_predict_2,
                verbose=True) == False
    w_message = str(w[0].message)
    m_message_1 = ('The \'fit\' method of the class has incorrect number '
                   '(1) of the required parameters. It needs to have exactly'
                   ' 2 required parameters. Try using optional '
                   'parameters if you require more functionality.')
    m_message_2 = ('The \'predict\' method of the class has incorrect number '
                   '(2) of the required parameters. It needs to have exactly'
                   ' 1 required parameters. Try using optional '
                   'parameters if you require more functionality.')
    assert (m_message_1 in w_message and
            m_message_2 in w_message)

    with pytest.warns(Warning) as w:
        assert fuv.check_model_functionality(class_fit_3_predict_0,
                                             verbose=True) == False
    w_message = str(w[0].message)
    m_message_1 = ('The \'fit\' method of the class has incorrect number '
                   '(3) of the required parameters. It needs to have exactly'
                   ' 2 required parameters. Try using optional '
                   'parameters if you require more functionality.')
    m_message_2 = ('The \'predict\' method of the class has incorrect number '
                   '(0) of the required parameters. It needs to have exactly'
                   ' 1 required parameters. Try using optional '
                   'parameters if you require more functionality.')
    assert (m_message_1 in w_message and
            m_message_2 in w_message)

    with pytest.warns(Warning) as w:
        assert fuv.check_model_functionality(class_fit_3_predict_0,
                                             True,
                                             True) == False
    w_message = str(w[0].message)
    m_message_1 = ('The \'fit\' method of the class has incorrect number '
                   '(3) of the required parameters. It needs to have exactly'
                   ' 2 required parameters. Try using optional '
                   'parameters if you require more functionality.')
    m_message_2 = ('The \'predict\' method of the class has incorrect number '
                   '(0) of the required parameters. It needs to have exactly'
                   ' 1 required parameters. Try using optional '
                   'parameters if you require more functionality.')
    assert (m_message_1 in w_message and
            m_message_2 in w_message and
            'missing \'predict_proba\'' in w_message)

    assert fuv.check_model_functionality(class_fit_2_predict_1_predict_proba_0)\
            == True
    assert fuv.check_model_functionality(class_fit_2_predict_1_predict_proba_0,
                                         True) == False
    assert fuv.check_model_functionality(class_fit_3_predict_1_predict_proba_0)\
            == False
    assert fuv.check_model_functionality(class_fit_3_predict_1_predict_proba_0,
                                         True) == False

    # Test predict_proba
    assert fuv.check_model_functionality(class_fit_2_predict_1) == True
    assert fuv.check_model_functionality(class_fit_2_predict_1, True) == False
    assert fuv.check_model_functionality(class_fit_2_predict_1_predict_proba_1,
                                        False,
                                        True) == True
    assert fuv.check_model_functionality(class_fit_2_predict_1_predict_proba_1,
                                         True,
                                         True) == True
