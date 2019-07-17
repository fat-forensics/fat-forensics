"""
Tests functions responsible for objects validation across FAT-Forensics.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
# License: new BSD

import pytest

import fatf.utils.validation as fuv


def test_check_explainer_functionality():
    """
    Tests :func:`fatf.utils.validation.check_explainer_functionality`.
    """
    class ClassPlain(object): pass
    class_plain = ClassPlain()
    class ClassInit(object):
        def __init__(self): pass
    class_init = ClassInit()
    class ClassExplainer1(object):
        def explain_instance(self): pass
    class_explainer_1 = ClassExplainer1()
    class ClassExplainer2(object):
        def explain_instance(self, x, y): pass
    class_explainer_2 = ClassExplainer2()
    class ClassExplainer3(object):
        def explain_instance(self, x): pass
    class_explainer_3 = ClassExplainer3()
    class ClassExplainer4(object):
        def explain_instance(self, x, y=3): pass
    class_explainer_4 = ClassExplainer4()
    class ClassExplainer5(object):
        def explain_instance(self, x, y=3, z=3): pass
    class_explainer_5 = ClassExplainer5()

    msg = ('The explainer class is missing \'explain_instance\' method.')
    with pytest.warns(UserWarning) as warning:
        assert fuv.check_explainer_functionality(class_plain, False) is False
    assert msg in str(warning[0].message)
    assert fuv.check_explainer_functionality(class_plain, True) is False

    with pytest.warns(UserWarning) as warning:
        assert fuv.check_explainer_functionality(class_init, False) is False
    assert msg in str(warning[0].message)
    assert fuv.check_explainer_functionality(class_init, True) is False

    msg = ('The \'explain_instance\' method of the class has incorrect number '
           '({}) of the required parameters. It needs to have exactly 1 '
           'required parameters. Try using optional parameters if you require '
           'more functionality.')
    with pytest.warns(UserWarning) as warning:
        assert fuv.check_explainer_functionality(
            class_explainer_1, False) is False
    assert msg.format(0) in str(warning[0].message)
    assert fuv.check_explainer_functionality(class_explainer_1, True) is False

    with pytest.warns(UserWarning) as warning:
        assert fuv.check_explainer_functionality(
            class_explainer_2, False) is False
    assert msg.format(2) in str(warning[0].message)
    assert fuv.check_explainer_functionality(class_explainer_2, True) is False

    assert fuv.check_explainer_functionality(class_explainer_3, False) is True
    assert fuv.check_explainer_functionality(class_explainer_4, False) is True
    assert fuv.check_explainer_functionality(class_explainer_5, False) is True


def test_check_kernel_functionality():
    """
    Tests :func:`fatf.utils.validation.check_kernel_functionality`.
    """
    error_msg = ('The \'{}\' kernel function has incorrect number '
                 '({}) of the required parameters. It needs to have '
                 'exactly 1 required parameters. Try using optional '
                 'parameters if you require more functionality.')

    def function1(): pass
    def function2(x): pass
    def function3(x, y): pass
    def function4(x, y=3): pass
    def function5(x=3, y=3): pass
    def function6(x, **kwargs): pass

    with pytest.warns(UserWarning) as warning:
        assert fuv.check_kernel_functionality(function1) is False
    assert str(warning[0].message) == error_msg.format('function1', 0)
    assert fuv.check_kernel_functionality(function1, True) is False

    with pytest.warns(UserWarning) as warning:
        assert fuv.check_kernel_functionality(function5) is False
    assert str(warning[0].message) == error_msg.format('function5', 0)
    assert fuv.check_kernel_functionality(function5, True) is False

    with pytest.warns(UserWarning) as warning:
        assert fuv.check_kernel_functionality(function3) is False
    assert str(warning[0].message) == error_msg.format('function3', 2)
    assert fuv.check_kernel_functionality(function3, True) is False

    assert fuv.check_kernel_functionality(function2) is True
    assert fuv.check_kernel_functionality(function4) is True
    assert fuv.check_kernel_functionality(function6) is True


def test_check_distance_functionality():
    """
    Tests :func:`fatf.utils.validation.check_distance_functionality`.
    """
    error_msg = ('The \'{}\' distance function has incorrect number '
                 '({}) of the required parameters. It needs to have '
                 'exactly 2 required parameters. Try using optional '
                 'parameters if you require more functionality.')

    def function1(): pass
    def function2(x): pass
    def function3(x, y): pass
    def function4(x, y, z=3): pass
    def function5(x=3, y=3): pass
    def function6(x, y, **kwargs): pass

    with pytest.warns(UserWarning) as warning:
        assert fuv.check_distance_functionality(function1) is False
    assert str(warning[0].message) == error_msg.format('function1', 0)
    assert fuv.check_distance_functionality(function1, True) is False

    with pytest.warns(UserWarning) as warning:
        assert fuv.check_distance_functionality(function5) is False
    assert str(warning[0].message) == error_msg.format('function5', 0)
    assert fuv.check_distance_functionality(function5, True) is False

    with pytest.warns(UserWarning) as warning:
        assert fuv.check_distance_functionality(function2) is False
    assert str(warning[0].message) == error_msg.format('function2', 1)
    assert fuv.check_distance_functionality(function2, True) is False

    assert fuv.check_distance_functionality(function3) is True
    assert fuv.check_distance_functionality(function4) is True
    assert fuv.check_distance_functionality(function6) is True
