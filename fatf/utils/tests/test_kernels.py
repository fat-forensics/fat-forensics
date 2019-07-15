"""
Tests kernel functions.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
# License: new BSD

import pytest

import numpy as np

from fatf.exceptions import IncorrectShapeError
import fatf.utils.kernels as fuk

# yapf: disable
CATEGORICAL_NP_ARRAY = np.array(['a', 'b', 'c'])
MIXED_ARRAY = np.array(
    [(0.1, 'a', 0.4, 'a')],
    dtype=[('a', 'i'), ('b', 'U1'), ('c', 'f'),('d', 'U2')])
NUMERICAL_NP_ARRAY = np.array([0.2, 0.1, 0.3, 0.5])
# yapf: enable


def test_input_is_valid():
    """
    Tests :func:`fatf.utils.kernels.__input_is_valid`.
    """
    structured_error = 'distances cannot be a structured array.'
    shape_error = 'distances must be a 1-dimensional array.'
    type_error = 'distances must be of numerical type.'
    with pytest.raises(TypeError) as exin:
        fuk._input_is_valid(MIXED_ARRAY)
    assert str(exin.value) == structured_error

    with pytest.raises(IncorrectShapeError) as exin:
        fuk._input_is_valid(np.ones((2, 2)))
    assert str(exin.value) == shape_error

    with pytest.raises(TypeError) as exin:
        fuk._input_is_valid(CATEGORICAL_NP_ARRAY)
    assert str(exin.value) == type_error

    # ALL FINE
    assert fuk._input_is_valid(NUMERICAL_NP_ARRAY)
    assert fuk._input_is_valid(np.ones((4,), dtype=np.int64))


def test_exponential_kernel():
    """
    Tests :func:`fatf.utils.kernels.exponential_kernel`.
    """
    width_type_err = 'width must be a float.'
    width_value_err = 'width must be a positive float greater than 0.'

    with pytest.raises(TypeError) as exin:
        fuk.exponential_kernel(NUMERICAL_NP_ARRAY, 1)
    assert str(exin.value) == width_type_err

    with pytest.raises(ValueError) as exin:
        fuk.exponential_kernel(NUMERICAL_NP_ARRAY, -0.1)
    assert str(exin.value) == width_value_err

    numerical_results = np.array([0.135, 0.607, 0.011, 0.00])
    kernel = fuk.exponential_kernel(NUMERICAL_NP_ARRAY, 0.1)
    assert np.allclose(kernel, numerical_results, atol=1e-3)

    numerical_results = np.array([0.923, 0.980, 0.835, 0.607])
    kernel = fuk.exponential_kernel(NUMERICAL_NP_ARRAY, 0.5)
    assert np.allclose(kernel, numerical_results, atol=1e-3)

    ones_results = np.array([0.607, 0.607, 0.607, 0.607])
    kernel = fuk.exponential_kernel(np.ones(4,), 1.0)
    assert np.allclose(kernel, ones_results, atol=1e-3)

    zeros_results = np.array([1.000, 1.000, 1.000, 1.000])
    kernel = fuk.exponential_kernel(np.zeros(4,), 1.0)
    assert np.allclose(kernel, zeros_results, atol=1e-3)
