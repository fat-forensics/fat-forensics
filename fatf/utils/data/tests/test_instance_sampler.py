"""
Tests instance samplers.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
#         Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import numpy as np

import pytest

import fatf
import fatf.utils.data.instance_sampler as fuis
from fatf.exceptions import IncorrectShapeError

# yapf: disable
NUMERICAL_NP_ARRAY = np.array([
    [0, 0, 0.08, 0.69],
    [1, 0, 0.03, 0.29],
    [0, 1, 0.99, 0.82],
    [2, 1, 0.73, 0.48],
    [1, 0, 0.36, 0.89],
    [0, 1, 0.07, 0.21]])
NUMERICAL_STRUCT_ARRAY = np.array(
    [(0, 0, 0.08, 0.69),
     (1, 0, 0.03, 0.29),
     (0, 1, 0.99, 0.82),
     (2, 1, 0.73, 0.48),
     (1, 0, 0.36, 0.89),
     (0, 1, 0.07, 0.21)],
    dtype=[('a', 'i'), ('b', 'i'), ('c', 'f'), ('d', 'f')])
CATEGORICAL_NP_ARRAY = np.array([
    ['a', 'b', 'c'],
    ['a', 'f', 'g'],
    ['b', 'c', 'c'],
    ['b', 'f', 'c'],
    ['a', 'f', 'c'],
    ['a', 'b', 'g']])
CATEGORICAL_STRUCT_ARRAY = np.array(
    [('a', 'b', 'c'),
     ('a', 'f', 'g'),
     ('b', 'c', 'c'),
     ('b', 'f', 'c'),
     ('a', 'f', 'c'),
     ('a', 'b', 'g')],
    dtype=[('a', 'U1'), ('b', 'U1'), ('c', 'U1')])
MIXED_ARRAY = np.array(
    [(0, 'a', 0.08, 'a'),
     (0, 'f', 0.03, 'bb'),
     (1, 'c', 0.99, 'aa'),
     (1, 'a', 0.73, 'a'),
     (0, 'c', 0.36, 'b'),
     (1, 'f', 0.07, 'bb')],
    dtype=[('a', 'i'), ('b', 'U1'), ('c', 'f'), ('d', 'U2')])
# yapf: enable


def test_validate_input():
    """
    Tests :func:`fatf.utils.data.instance_sampler._validate_input`.
    """
    shape_msg = ('The data_row must either be a 1-dimensional numpy array or '
                 'numpy void object for structured rows.')
    samples_msg  = 'The samples_number parameter must be a positive integer.'
    samples_type_msg = 'The samples_number parameter must be an integer.'

    with pytest.raises(IncorrectShapeError) as exin:
        fuis._validate_input(NUMERICAL_NP_ARRAY, 50)
    assert str(exin.value) == shape_msg

    with pytest.raises(ValueError) as exin:
        fuis._validate_input(NUMERICAL_NP_ARRAY[0], -1)
    assert str(exin.value) == samples_msg

    with pytest.raises(TypeError) as exin:
        fuis._validate_input(NUMERICAL_NP_ARRAY[0], 'a')
    assert str(exin.value) == samples_type_msg

    # Everything good
    assert fuis._validate_input(NUMERICAL_NP_ARRAY[0], 50)
    assert fuis._validate_input(CATEGORICAL_NP_ARRAY[0], 50)
    assert fuis._validate_input(NUMERICAL_STRUCT_ARRAY[0], 50)
    assert fuis._validate_input(CATEGORICAL_STRUCT_ARRAY[0], 50)
    assert fuis._validate_input(MIXED_ARRAY[0], 50)


def test_instance_sampler():
    """
    Tests :func:`fatf.utils.data.instance_sampler.binary_sampler`.
    """
    fatf.setup_random_seed()

    numerical_binary_array = np.array([1, 0, 1, 1])
    numerical_binary_array_sampled = np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 1, 0]])

    numerical_binary_struct_array = np.array([(1, 0, 1., 1.)],
        dtype=[('a', 'i'), ('b', 'i'), ('c', 'f'), ('d', 'f')])
    numerical_binary_struct_array_sampled = np.array(
        [(1, 0, 0., 0.),
         (0, 0, 0., 1.),
         (1, 0, 0., 1.),
         (1, 0, 1., 1.),
         (1, 0, 0., 0.)],
        dtype=[('a', 'i'), ('b', 'i'), ('c', 'f'), ('d', 'f')])

    proportions = [0.5, 0., 0.5, 0.5]

    binary_msg = 'data_row is not a binary row.'

    with pytest.raises(ValueError) as exin:
        samples = fuis.binary_sampler(np.array([0, 1, 2, 3]))
    assert str(exin.value) == binary_msg

    with pytest.raises(ValueError) as exin:
        samples = fuis.binary_sampler(np.array([0., 0.5, 0.5, 0.2]))
    assert str(exin.value) == binary_msg

    with pytest.raises(ValueError) as exin:
        samples = fuis.binary_sampler(CATEGORICAL_STRUCT_ARRAY[0])
    assert str(exin.value) == binary_msg

    with pytest.raises(ValueError) as exin:
        samples = fuis.binary_sampler(MIXED_ARRAY[0])
    assert str(exin.value) == binary_msg

    samples = fuis.binary_sampler(numerical_binary_array, samples_number=5)
    assert np.array_equal(samples, numerical_binary_array_sampled)

    samples = fuis.binary_sampler(numerical_binary_array, samples_number=1000)
    assert np.allclose(samples.sum(axis=0)/samples.shape[0],
                       np.array(proportions), atol=1e-1)

    samples = fuis.binary_sampler(numerical_binary_struct_array[0],
                                  samples_number=5)
    assert np.array_equal(samples, numerical_binary_struct_array_sampled)

    samples = fuis.binary_sampler(numerical_binary_struct_array[0],
                                  samples_number=1000)
    for i, name in enumerate(numerical_binary_struct_array.dtype.names):
        assert np.allclose(samples[name].sum()/samples[name].shape[0],
                           proportions[i], atol=1e-1)
