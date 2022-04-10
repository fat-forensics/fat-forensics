"""
Tests instance augmentation functions.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
#         Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import numpy as np

import pytest

import fatf
import fatf.utils.array.validation as fuav
import fatf.utils.data.instance_augmentation as fudi

from fatf.exceptions import IncorrectShapeError

# yapf: disable
NUMERICAL_NP_ARRAY = np.array([[0, 0, 0.08, 0.69], [0, 1, 0.07, 0.21]])
NUMERICAL_STRUCT_ARRAY = np.array(
    [(0, 0, 0.08, 0.69), (1, 0, 0.03, 0.29)],
    dtype=[('a', 'i'), ('b', 'i'), ('c', 'f'), ('d', 'f')]
)
CATEGORICAL_NP_ARRAY = np.array([['a', 'b', 'c'], ['a', 'f', 'g']])
CATEGORICAL_STRUCT_ARRAY = np.array(
    [('a', 'b', 'c'), ('a', 'f', 'g')],
    dtype=[('a', 'U1'), ('b', 'U1'), ('c', 'U1')]
)
MIXED_ARRAY = np.array(
    [(0, 'a', 0.08, 'a'), (0, 'f', 0.03, 'bb')],
    dtype=[('a', 'i'), ('b', 'U1'), ('c', 'f'), ('d', 'U2')])
# yapf: enable


def test_validate_input():
    """
    Tests :func:`fatf.utils.data.instance_augmentation._validate_input`.
    """
    shape_msg = ('The data_row must either be a 1-dimensional numpy array or '
                 'a numpy void object for structured rows.')
    samples_msg = 'The samples_number parameter must be a positive integer.'
    samples_type_msg = 'The samples_number parameter must be an integer.'

    with pytest.raises(IncorrectShapeError) as exin:
        fudi._validate_input(NUMERICAL_NP_ARRAY, None)
    assert str(exin.value) == shape_msg
    with pytest.raises(IncorrectShapeError) as exin:
        fudi._validate_input(NUMERICAL_STRUCT_ARRAY, None)
    assert str(exin.value) == shape_msg

    with pytest.raises(ValueError) as exin:
        fudi._validate_input(NUMERICAL_NP_ARRAY[0], -1)
    assert str(exin.value) == samples_msg
    with pytest.raises(ValueError) as exin:
        fudi._validate_input(NUMERICAL_STRUCT_ARRAY[0], 0)
    assert str(exin.value) == samples_msg

    with pytest.raises(TypeError) as exin:
        fudi._validate_input(NUMERICAL_NP_ARRAY[0], 'a')
    assert str(exin.value) == samples_type_msg
    with pytest.raises(TypeError) as exin:
        fudi._validate_input(NUMERICAL_STRUCT_ARRAY[0], 40.)
    assert str(exin.value) == samples_type_msg

    # Everything OK
    assert fudi._validate_input(NUMERICAL_NP_ARRAY[0], 50)
    assert fudi._validate_input(NUMERICAL_STRUCT_ARRAY[0], 50)
    assert fudi._validate_input(CATEGORICAL_NP_ARRAY[0], 50)
    assert fudi._validate_input(CATEGORICAL_STRUCT_ARRAY[0], 50)
    assert fudi._validate_input(MIXED_ARRAY[0], 50)


def test_binary_sampler():
    """
    Tests :func:`fatf.utils.data.instance_augmentation.binary_sampler`.
    """
    fatf.setup_random_seed()

    binary_msg = 'The data_row is not binary.'
    proportions = [0.5, 0., 0.5, 0.5]

    numerical_binary_array = np.array([1, 0, 1, 1])
    numerical_binary_array_sampled = np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ])  # yapf: disable

    struct_dtype = [('a', 'i'), ('b', 'i'), ('c', 'f'), ('d', bool)]
    numerical_binary_struct_array = np.array([(1, 0, 1., True)],
                                             dtype=struct_dtype)
    numerical_binary_struct_array = numerical_binary_struct_array[0]
    numerical_binary_struct_array_sampled = np.array(
        [(1, 0, 0., False),
         (0, 0, 0., True),
         (1, 0, 0., True),
         (1, 0, 1., True),
         (1, 0, 0., False)],
        dtype=struct_dtype)  # yapf: disable

    with pytest.raises(ValueError) as exin:
        fudi.binary_sampler(np.array([0, 1, 2, 3]))
    assert str(exin.value) == binary_msg
    with pytest.raises(ValueError) as exin:
        fudi.binary_sampler(np.array([0., 0.5, 0.5, 0.2]))
    assert str(exin.value) == binary_msg
    with pytest.raises(ValueError) as exin:
        fudi.binary_sampler(CATEGORICAL_STRUCT_ARRAY[0])
    assert str(exin.value) == binary_msg
    with pytest.raises(ValueError) as exin:
        fudi.binary_sampler(MIXED_ARRAY[0])
    assert str(exin.value) == binary_msg

    #

    samples = fudi.binary_sampler(numerical_binary_array, samples_number=5)
    assert np.array_equal(samples, numerical_binary_array_sampled)

    samples = fudi.binary_sampler(numerical_binary_array, samples_number=1000)
    assert np.allclose(
        samples.sum(axis=0) / samples.shape[0], proportions, atol=1e-1)

    samples = fudi.binary_sampler(
        numerical_binary_struct_array, samples_number=5)
    assert np.array_equal(samples, numerical_binary_struct_array_sampled)
    assert fuav.are_similar_dtype_arrays(
        np.asarray(numerical_binary_struct_array), samples, True)

    samples = fudi.binary_sampler(
        numerical_binary_struct_array, samples_number=1000)
    for i, name in enumerate(numerical_binary_struct_array.dtype.names):
        assert np.allclose(
            samples[name].sum() / samples[name].shape[0],
            proportions[i],
            atol=1e-1)
    assert fuav.are_similar_dtype_arrays(
        np.asarray(numerical_binary_struct_array), samples, True)


def test_random_binary_sampler():
    """
    Tests :func:`fatf.utils.data.instance_augmentation.random_binary_sampler`.
    """
    err_msg = 'The number of elements must be an integer.'
    with pytest.raises(TypeError) as exin:
        fudi.random_binary_sampler('int')
    assert str(exin.value) == err_msg
    with pytest.raises(TypeError) as exin:
        fudi.random_binary_sampler(1.0)
    assert str(exin.value) == err_msg

    err_msg = 'The number of elements must be greater than 0.'
    with pytest.raises(ValueError) as exin:
        fudi.random_binary_sampler(0)
    assert str(exin.value) == err_msg
    with pytest.raises(ValueError) as exin:
        fudi.random_binary_sampler(-42)
    assert str(exin.value) == err_msg

    err_msg = 'The number of samples must be an integer.'
    with pytest.raises(TypeError) as exin:
        fudi.random_binary_sampler(4, 'int')
    assert str(exin.value) == err_msg
    with pytest.raises(TypeError) as exin:
        fudi.random_binary_sampler(4, 4.2)
    assert str(exin.value) == err_msg

    err_msg = 'The number of samples must be greater than 0.'
    with pytest.raises(ValueError) as exin:
        fudi.random_binary_sampler(4, 0)
    assert str(exin.value) == err_msg
    with pytest.raises(ValueError) as exin:
        fudi.random_binary_sampler(4, -42)
    assert str(exin.value) == err_msg

    fatf.setup_random_seed()
    sample = fudi.random_binary_sampler(4, 10)
    sample_ = np.array([[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                        [1, 1, 1, 0], [1, 0, 1, 1], [1, 1, 1, 1], [1, 1, 0, 0],
                        [1, 1, 1, 0], [1, 0, 0, 0]])
    assert np.array_equal(sample, sample_)
