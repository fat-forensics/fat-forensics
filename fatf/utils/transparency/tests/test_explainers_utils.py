"""
Tests explainers utilities.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import pytest

import fatf.utils.transparency.explainers as fute


class TestExplainer(object):
    """
    Tests :class:`fatf.utils.transparency.explainers.Explainer` class.
    """
    explainer = fute.Explainer()

    def test_explainer_class(self):
        """
        Tests ``Explainer`` class initialisation.
        """
        assert self.explainer.__class__.__bases__[0].__name__ == 'ABC'
        assert self.explainer.__class__.__name__ == 'Explainer'

    def test_explainer_class_errors(self):
        """
        Tests ``Explainer``'s unimplemented methods.
        """
        feature_importance = 'Feature importance not implemented.'
        model_explanation = 'Model explanation (global) not implemented.'
        instance_explanation = ('Data point explanation (local) not '
                                'implemented.')

        with pytest.raises(NotImplementedError) as exinf:
            self.explainer.feature_importance()
        assert str(exinf.value) == feature_importance
        with pytest.raises(NotImplementedError) as exinf:
            self.explainer.explain_model()
        assert str(exinf.value) == model_explanation
        with pytest.raises(NotImplementedError) as exinf:
            self.explainer.explain_instance()
        assert str(exinf.value) == instance_explanation


def test_check_instance_explainer_functionality():
    """
    Tests the ``check_instance_explainer_functionality`` function.

    This test is for the :func:`fatf.utils.transparency.explainers.\
check_instance_explainer_functionality` function.
    """
    type_error = 'The suppress_warning parameter should be a boolean.'
    inheritance_warning = (
        'Every explainer object should inherit from fatf.utils.transparency.'
        'explainers.Explainer abstract class.')

    class ClassPlain(object):
        pass

    class_plain = ClassPlain()

    class ClassInit(fute.Explainer):
        def __init__(self):
            pass

    class_init = ClassInit()

    class ClassExplainer1(object):
        def explain_instance(self):
            pass  # pragma: no cover

    class_explainer_1 = ClassExplainer1()

    class ClassExplainer2(fute.Explainer):
        def explain_instance(self, x, y):
            pass  # pragma: no cover

    class_explainer_2 = ClassExplainer2()

    class ClassExplainer3(object):
        def explain_instance(self, x):
            pass  # pragma: no cover

    class_explainer_3 = ClassExplainer3()

    class ClassExplainer4(fute.Explainer):
        def explain_instance(self, x, y=3):
            pass  # pragma: no cover

    class_explainer_4 = ClassExplainer4()

    class ClassExplainer5(object):
        def explain_instance(self, x, y=3, z=3):
            pass  # pragma: no cover

    class_explainer_5 = ClassExplainer5()

    with pytest.raises(TypeError) as exinf:
        fute.check_instance_explainer_functionality(class_plain, 'False')
    assert str(exinf.value) == type_error
    with pytest.raises(TypeError) as exinf:
        fute.check_instance_explainer_functionality(ClassPlain, 'True')
    assert str(exinf.value) == type_error

    msg = "The *{}* (explainer) class is missing 'explain_instance' method."

    with pytest.warns(UserWarning) as warning:
        assert fute.check_instance_explainer_functionality(class_plain,
                                                           False) is False
    assert len(warning) == 2
    assert msg.format('ClassPlain') == str(warning[0].message)
    assert inheritance_warning == str(warning[1].message)

    with pytest.warns(UserWarning) as warning:
        assert fute.check_instance_explainer_functionality(ClassPlain) is False
    assert len(warning) == 2
    assert msg.format('ClassPlain') == str(warning[0].message)
    assert inheritance_warning == str(warning[1].message)

    with pytest.warns(UserWarning) as warning:
        assert fute.check_instance_explainer_functionality(class_plain,
                                                           True) is False
    assert len(warning) == 1
    assert inheritance_warning == str(warning[0].message)

    msg = ("The 'explain_instance' method of the *{}* (explainer) class has "
           'incorrect number ({}) of the required parameters. It needs to '
           'have exactly 1 required parameter(s). Try using optional '
           'parameters if you require more functionality.')

    with pytest.warns(UserWarning) as warning:
        assert fute.check_instance_explainer_functionality(class_init,
                                                           False) is False
    assert len(warning) == 1
    assert msg.format('ClassInit', 0) == str(warning[0].message)

    with pytest.warns(UserWarning) as warning:
        assert fute.check_instance_explainer_functionality(ClassInit) is False
    assert len(warning) == 1
    assert msg.format('ClassInit', 0) == str(warning[0].message)

    assert fute.check_instance_explainer_functionality(class_init,
                                                       True) is False

    #

    with pytest.warns(UserWarning) as warning:
        assert fute.check_instance_explainer_functionality(
            class_explainer_1, False) is False
    assert len(warning) == 2
    assert msg.format('ClassExplainer1', 0) == str(warning[0].message)
    assert inheritance_warning == str(warning[1].message)

    with pytest.warns(UserWarning) as warning:
        assert fute.check_instance_explainer_functionality(
            ClassExplainer1) is False
    assert len(warning) == 2
    assert msg.format('ClassExplainer1', 0) == str(warning[0].message)
    assert inheritance_warning == str(warning[1].message)

    with pytest.warns(UserWarning) as warning:
        assert fute.check_instance_explainer_functionality(
            class_explainer_1, True) is False
    assert len(warning) == 1
    assert inheritance_warning == str(warning[0].message)

    #

    with pytest.warns(UserWarning) as warning:
        assert fute.check_instance_explainer_functionality(
            class_explainer_2, False) is False
    assert len(warning) == 1
    assert msg.format('ClassExplainer2', 2) == str(warning[0].message)

    with pytest.warns(UserWarning) as warning:
        assert fute.check_instance_explainer_functionality(
            class_explainer_2) is False
    assert len(warning) == 1
    assert msg.format('ClassExplainer2', 2) == str(warning[0].message)

    assert fute.check_instance_explainer_functionality(class_explainer_2,
                                                       True) is False

    #
    #

    with pytest.warns(UserWarning) as warning:
        assert fute.check_instance_explainer_functionality(
            class_explainer_3, False) is True
    assert len(warning) == 1
    assert inheritance_warning == str(warning[0].message)

    with pytest.warns(UserWarning) as warning:
        assert fute.check_instance_explainer_functionality(
            ClassExplainer3, True) is True
    assert len(warning) == 1
    assert inheritance_warning == str(warning[0].message)

    #

    assert fute.check_instance_explainer_functionality(class_explainer_4,
                                                       False) is True
    assert fute.check_instance_explainer_functionality(ClassExplainer4,
                                                       True) is True

    #

    with pytest.warns(UserWarning) as warning:
        assert fute.check_instance_explainer_functionality(
            class_explainer_5, False) is True
    assert len(warning) == 1
    assert inheritance_warning == str(warning[0].message)

    with pytest.warns(UserWarning) as warning:
        assert fute.check_instance_explainer_functionality(
            ClassExplainer5, True) is True
    assert len(warning) == 1
    assert inheritance_warning == str(warning[0].message)
