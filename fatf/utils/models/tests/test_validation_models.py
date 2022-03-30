"""
Tests functions responsible for objects validation across FAT Forensics.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import pytest

import fatf.utils.models.validation as fumv


def test_check_model_functionality():
    """
    Tests :func:`fatf.utils.models.validation.check_model_functionality`.
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

    # Test errors
    require_probabilities_error = ('The require_probabilities parameter must '
                                   'be boolean.')
    suppress_warning_error = 'The suppress_warning parameter must be boolean.'
    with pytest.raises(TypeError) as error:
        fumv.check_model_functionality(
            class_plain, require_probabilities='true')
    assert str(error.value) == require_probabilities_error
    with pytest.raises(TypeError) as error:
        fumv.check_model_functionality(
            class_plain, require_probabilities='true', suppress_warning='true')
    assert str(error.value) == require_probabilities_error
    with pytest.raises(TypeError) as error:
        fumv.check_model_functionality(
            class_plain, require_probabilities=True, suppress_warning='true')
    assert str(error.value) == suppress_warning_error

    # Test not suppressed -- warning
    with pytest.warns(UserWarning) as warning:
        assert fumv.check_model_functionality(class_plain, True,
                                              False) is False
    w_message = str(warning[0].message)
    assert ("missing 'fit'" in w_message and "missing 'predict'" in w_message
            and "missing 'predict_proba'" in w_message)

    # Test suppressed -- warning
    assert fumv.check_model_functionality(class_plain, True, True) is False

    # Test optional arguments
    assert fumv.check_model_functionality(
        class_fit_11_predict_1, suppress_warning=True) is False
    assert fumv.check_model_functionality(class_fit_21_predict_1) is True

    # Too few method parameters
    with pytest.warns(UserWarning) as warning:
        assert fumv.check_model_functionality(
            class_fit_1_predict_2, suppress_warning=False) is False
    w_message = str(warning[0].message)
    m_message_1 = ("The 'fit' method of the *ClassFit1Predict2* (model) class "
                   'has incorrect number (1) of the required parameters. It '
                   'needs to have exactly 2 required parameter(s). Try using '
                   'optional parameters if you require more functionality.')
    m_message_2 = ("The 'predict' method of the *ClassFit1Predict2* (model) "
                   'class has incorrect number (2) of the required '
                   'parameters. It needs to have exactly 1 required '
                   'parameter(s). Try using optional parameters if you '
                   'require more functionality.')
    assert m_message_1 in w_message and m_message_2 in w_message

    # Not an instance
    with pytest.warns(UserWarning) as warning:
        assert fumv.check_model_functionality(
            ClassFit1Predict2, suppress_warning=False) is False
    w_message = str(warning[0].message)
    assert m_message_1 in w_message and m_message_2 in w_message

    with pytest.warns(UserWarning) as warning:
        assert fumv.check_model_functionality(
            class_fit_3_predict_0, suppress_warning=False) is False
    w_message = str(warning[0].message)
    m_message_1 = ("The 'fit' method of the *ClassFit3Predict0* (model) class "
                   'has incorrect number (3) of the required parameters. It '
                   'needs to have exactly 2 required parameter(s). Try using '
                   'optional parameters if you require more functionality.')
    m_message_2 = ("The 'predict' method of the *ClassFit3Predict0* (model) "
                   'class has incorrect number (0) of the required '
                   'parameters. It needs to have exactly 1 required '
                   'parameter(s). Try using optional parameters if you '
                   'require more functionality.')
    assert m_message_1 in w_message and m_message_2 in w_message

    # Not an instance
    with pytest.warns(UserWarning) as warning:
        assert fumv.check_model_functionality(
            ClassFit3Predict0, suppress_warning=False) is False
    assert m_message_1 in w_message and m_message_2 in w_message

    with pytest.warns(UserWarning) as warning:
        assert fumv.check_model_functionality(class_fit_3_predict_0, True,
                                              False) is False
    w_message = str(warning[0].message)
    m_message_1 = ("The 'fit' method of the *ClassFit3Predict0* (model) class "
                   'has incorrect number (3) of the required parameters. It '
                   'needs to have exactly 2 required parameter(s). Try using '
                   'optional parameters if you require more functionality.')
    m_message_2 = ("The 'predict' method of the *ClassFit3Predict0* (model) "
                   'class has incorrect number (0) of the required '
                   'parameters. It needs to have exactly 1 required '
                   'parameter(s). Try using optional parameters if you '
                   'require more functionality.')
    assert (m_message_1 in w_message and m_message_2 in w_message
            and 'missing \'predict_proba\'' in w_message)

    # Not an instance
    with pytest.warns(UserWarning) as warning:
        assert fumv.check_model_functionality(ClassFit3Predict0, True,
                                              False) is False
    w_message = str(warning[0].message)

    assert fumv.check_model_functionality(
        class_fit_2_predict_1_predict_proba_0) is True
    assert fumv.check_model_functionality(
        ClassFit2Predict1PredictProba0) is True
    #
    assert fumv.check_model_functionality(
        class_fit_2_predict_1_predict_proba_0, True,
        suppress_warning=True) is False
    assert fumv.check_model_functionality(
        ClassFit2Predict1PredictProba0, True, suppress_warning=True) is False
    #
    #
    assert fumv.check_model_functionality(
        class_fit_3_predict_1_predict_proba_0, suppress_warning=True) is False
    assert fumv.check_model_functionality(
        ClassFit3Predict1PredictProba0, suppress_warning=True) is False
    #
    assert fumv.check_model_functionality(
        class_fit_3_predict_1_predict_proba_0, True,
        suppress_warning=True) is False
    assert fumv.check_model_functionality(
        ClassFit3Predict1PredictProba0, True, suppress_warning=True) is False

    # Test predict_proba
    assert fumv.check_model_functionality(class_fit_2_predict_1) is True
    assert fumv.check_model_functionality(ClassFit2Predict1) is True
    #
    assert fumv.check_model_functionality(
        class_fit_2_predict_1, True, suppress_warning=True) is False
    assert fumv.check_model_functionality(
        ClassFit2Predict1, True, suppress_warning=True) is False
    ##
    assert fumv.check_model_functionality(
        class_fit_2_predict_1_predict_proba_1, False) is True
    assert fumv.check_model_functionality(ClassFit2Predict1PredictProba1,
                                          False) is True
    #
    assert fumv.check_model_functionality(
        class_fit_2_predict_1_predict_proba_1, True) is True
    assert fumv.check_model_functionality(ClassFit2Predict1PredictProba1,
                                          True) is True
