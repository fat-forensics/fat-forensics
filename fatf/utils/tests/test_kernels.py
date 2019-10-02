"""
Tests implementations of kernel functions.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
#         Kacper Sokol <K.Sokol@bristol.ac.uk>
# License: new BSD

import pytest

import numpy as np

import fatf.utils.kernels as fatf_kernels

from fatf.exceptions import IncorrectShapeError

NUMERICAL_NP_ARRAY = np.array([0.2, 0.1, 0.3, 0.5])
CATEGORICAL_NP_ARRAY = np.array(['a', 'b', 'c'])
MIXED_ARRAY = np.array(
    [(0.1, 'a', 0.4, 'a')],
    dtype=[('a', 'i'), ('b', 'U1'), ('c', 'f'), ('d', 'U2')])  # yapf: disable


def test_input_is_valid():
    """
    Tests :func:`fatf.utils.kernels._input_is_valid` function.
    """
    structured_error = 'The distances array cannot be a structured array.'
    shape_error = 'The distances array must be a 1-dimensional array.'
    type_error = 'The distances array must be of numerical type.'

    with pytest.raises(TypeError) as exin:
        fatf_kernels._input_is_valid(MIXED_ARRAY)
    assert str(exin.value) == structured_error

    with pytest.raises(IncorrectShapeError) as exin:
        fatf_kernels._input_is_valid(np.ones((2, 2)))
    assert str(exin.value) == shape_error

    with pytest.raises(TypeError) as exin:
        fatf_kernels._input_is_valid(CATEGORICAL_NP_ARRAY)
    assert str(exin.value) == type_error

    # All inputs valid
    assert fatf_kernels._input_is_valid(NUMERICAL_NP_ARRAY)
    assert fatf_kernels._input_is_valid(np.ones((4, ), dtype=np.int64))


def test_exponential_kernel():
    """
    Tests :func:`fatf.utils.kernels.exponential_kernel` function.
    """
    width_type_err = 'The kernel width must be a number.'
    width_value_err = ('The kernel width must be a positive (greater than 0) '
                       'number.')

    with pytest.raises(TypeError) as exin:
        fatf_kernels.exponential_kernel(NUMERICAL_NP_ARRAY, '1')
    assert str(exin.value) == width_type_err

    with pytest.raises(ValueError) as exin:
        fatf_kernels.exponential_kernel(NUMERICAL_NP_ARRAY, -0.1)
    assert str(exin.value) == width_value_err

    with pytest.raises(ValueError) as exin:
        fatf_kernels.exponential_kernel(NUMERICAL_NP_ARRAY, 0)
    assert str(exin.value) == width_value_err

    results = np.array([0.135, 0.607, 0.011, 0])
    kernelised = fatf_kernels.exponential_kernel(NUMERICAL_NP_ARRAY, 0.1)
    assert np.allclose(kernelised, results, atol=1e-3)

    results = np.array([0.923, 0.980, 0.835, 0.607])
    kernelised = fatf_kernels.exponential_kernel(NUMERICAL_NP_ARRAY, 0.5)
    assert np.allclose(kernelised, results, atol=1e-3)

    results = np.array([0.607, 0.607, 0.607, 0.607])
    kernelised = fatf_kernels.exponential_kernel(np.ones(4, ), 1.0)
    assert np.allclose(kernelised, results, atol=1e-3)

    results = np.array([1, 1, 1, 1])
    kernelised = fatf_kernels.exponential_kernel(np.zeros(4), 1)
    assert np.allclose(kernelised, results, atol=1e-3)


def test_check_kernel_functionality():
    """
    Tests :func:`fatf.utils.kernels.check_kernel_functionality` function.
    """
    kernel_type_error = ('The kernel_function parameter should be a Python '
                         'callable.')
    suppress_type_error = 'The suppress_warning parameter should be a boolean.'

    error_msg = ("The '{}' kernel function has incorrect number "
                 '({}) of the required parameters. It needs to have '
                 'exactly 1 required parameter. Try using optional '
                 'parameters if you require more functionality.')

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

    with pytest.raises(TypeError) as exin:
        fatf_kernels.check_kernel_functionality('callable')
    assert str(exin.value) == kernel_type_error
    with pytest.raises(TypeError) as exin:
        fatf_kernels.check_kernel_functionality('callable', 'True')
    assert str(exin.value) == kernel_type_error
    with pytest.raises(TypeError) as exin:
        fatf_kernels.check_kernel_functionality(function1, 'True')
    assert str(exin.value) == suppress_type_error

    with pytest.warns(UserWarning) as warning:
        assert fatf_kernels.check_kernel_functionality(function1) is False
    assert len(warning) == 1
    assert str(warning[0].message) == error_msg.format('function1', 0)
    #
    with pytest.warns(UserWarning) as warning:
        assert fatf_kernels.check_kernel_functionality(function1,
                                                       False) is False
    assert len(warning) == 1
    assert str(warning[0].message) == error_msg.format('function1', 0)
    #
    assert fatf_kernels.check_kernel_functionality(function1, True) is False

    #

    with pytest.warns(UserWarning) as warning:
        assert fatf_kernels.check_kernel_functionality(function5) is False
    assert len(warning) == 1
    assert str(warning[0].message) == error_msg.format('function5', 0)
    #
    with pytest.warns(UserWarning) as warning:
        assert fatf_kernels.check_kernel_functionality(function5,
                                                       False) is False
    assert len(warning) == 1
    assert str(warning[0].message) == error_msg.format('function5', 0)
    #
    assert fatf_kernels.check_kernel_functionality(function5, True) is False

    #

    with pytest.warns(UserWarning) as warning:
        assert fatf_kernels.check_kernel_functionality(function3) is False
    assert len(warning) == 1
    assert str(warning[0].message) == error_msg.format('function3', 2)
    #
    with pytest.warns(UserWarning) as warning:
        assert fatf_kernels.check_kernel_functionality(function3,
                                                       False) is False
    assert len(warning) == 1
    assert str(warning[0].message) == error_msg.format('function3', 2)
    #
    assert fatf_kernels.check_kernel_functionality(function3, True) is False

    #
    #

    assert fatf_kernels.check_kernel_functionality(function2) is True
    assert fatf_kernels.check_kernel_functionality(function4, False) is True
    assert fatf_kernels.check_kernel_functionality(function6, True) is True
