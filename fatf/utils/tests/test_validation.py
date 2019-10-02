"""
Tests functions responsible for generic objects and functions validation.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
#         Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import pytest

import fatf.utils.validation as fuv


def test_get_required_parameters_number():
    """
    Tests :func:`fatf.utils.validation.get_required_parameters_number`.
    """
    type_error = ('The callable_object should be Python callable, e.g., a '
                  'function or a method.')

    with pytest.raises(TypeError) as error:
        fuv.get_required_parameters_number('callable')
    assert str(error.value) == type_error

    def function1():
        pass  # pragma: no cover

    def function2(x):
        pass  # pragma: no cover

    def function3(x, y):
        pass  # pragma: no cover

    def function4(x, y=3):
        pass  # pragma: no cover

    def function5(x=3, y=3):
        pass  # pragma: no cover

    def function6(x, **kwargs):
        pass  # pragma: no cover

    assert fuv.get_required_parameters_number(function1) == 0
    assert fuv.get_required_parameters_number(function2) == 1
    assert fuv.get_required_parameters_number(function3) == 2
    assert fuv.get_required_parameters_number(function4) == 1
    assert fuv.get_required_parameters_number(function5) == 0
    assert fuv.get_required_parameters_number(function6) == 1


def test_check_object_functionality():
    """
    Tests :func:`fatf.utils.validation.check_object_functionality` function.
    """
    methods_key_type_error = ('All of the keys in the methods dictionary must '
                              "be strings. The '{}' key in not a string.")
    methods_value_type_error = ('All of the values in the methods dictionary '
                                "must be integers. The '{}' value for the "
                                "'{}' key in not a string.")
    methods_value_value_error = ('All of the values in the methods dictionary '
                                 "must be non-negative integers. The '{}' "
                                 "value for '{}' key does not comply.")
    methods_empty_value_error = 'The methods dictionary cannot be empty.'
    methods_type_error = 'The methods parameter must be a dictionary.'
    #
    reference_type_error = ('The object_reference_name parameter must be a '
                            'string or None.')

    missing_callable = "The {} is missing '{}' method."
    missing_param = ("The '{}' method of the {} has incorrect number "
                     '({}) of the required parameters. It needs to have '
                     'exactly {} required parameter(s). Try using optional '
                     'parameters if you require more functionality.')

    with pytest.raises(TypeError) as exin:
        fuv.check_object_functionality('object', 'dict', 574)
    assert str(exin.value) == methods_type_error
    #
    with pytest.raises(ValueError) as exin:
        fuv.check_object_functionality('object', {}, 574)
    assert str(exin.value) == methods_empty_value_error
    #
    with pytest.raises(TypeError) as exin:
        fuv.check_object_functionality('object', {'1': 1, 2: '2', '3': 3}, 574)
    assert str(exin.value) == methods_key_type_error.format(2)
    #
    with pytest.raises(TypeError) as exin:
        fuv.check_object_functionality(
            'object', {'1': '1', 2: '2', '3': 3}, 574)  # yapf: disable
    assert str(exin.value) == methods_value_type_error.format(1, 1)
    #
    with pytest.raises(ValueError) as exin:
        fuv.check_object_functionality(
            'object', {'1': 1, '2': -2, '3': 3}, 574)  # yapf: disable
    assert str(exin.value) == methods_value_value_error.format(-2, 2)
    #
    #
    with pytest.raises(TypeError) as exin:
        fuv.check_object_functionality('object', {'1': 1, '2': 2, '3': 3}, 574)
    assert str(exin.value) == reference_type_error

    class A(object):
        pass

    class B(A):
        def zero(self):
            pass  # pragma: no cover

        def one(self, a, b=0):
            pass  # pragma: no cover

    class C(B):
        def zero(self, a=0, b=1):
            pass  # pragma: no cover

    b = B()
    c = C()

    is_functional, msg = fuv.check_object_functionality(b, {'one': 1})
    assert is_functional
    assert msg == ''

    is_functional, msg = fuv.check_object_functionality(
        B, {'one': 1, 'zero': 0}, None)  # yapf: disable
    assert is_functional
    assert msg == ''

    is_functional, msg = fuv.check_object_functionality(
        c, {'one': 1, 'zero': 0, 'two': 2}, 'test object')  # yapf: disable
    assert not is_functional
    assert msg == missing_callable.format('*C* (test object) class', 'two')

    is_functional, msg = fuv.check_object_functionality(
        C, {'one': 1, 'zero': 2, 'two': 2}, None)  # yapf: disable
    assert not is_functional
    assert missing_callable.format('*C* class', 'two') in msg
    assert missing_param.format('zero', '*C* class', 0, 2) in msg
