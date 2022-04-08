"""
Tets the :mod:`fatf.transparency.models.submodular_pick` module.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
#         Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import pytest

import numpy as np

from fatf.exceptions import IncorrectShapeError
from fatf.utils.testing.arrays import NOT_BASE_NP_ARRAY

import fatf

import fatf.transparency.models.submodular_pick as ftms

# yapf: disable
NUMERICAL_NP_ARRAY = np.array([
    [0, 0, 0.08, 0.69],
    [1, 0, 0.03, 0.29],
    [0, 1, 0.99, 0.82],
    [0, 1, 0.07, 0.21]])

CATEGORICAL_STRUCT_ARRAY = np.array(
    [('a', 'b', 'c'),
     ('a', 'f', 'g'),
     ('b', 'c', 'c')],
    dtype=[('a', 'U1'), ('b', 'U1'), ('c', 'U1')])

EXPLAINERS = [
    {'a': 0.1, 'b': 0.1, 'c': 0.1, 'd': 0.1},
    {'a': 0.1, 'b': 0.1},
    {'e': 0.1, 'd': 0.1},
    {'a': 0.9, 'b': 0.9, 'c': 0.9, 'd': 0.9}]
# yapf: enable


def explain_instance_a(instance):
    ind = np.where((NUMERICAL_NP_ARRAY == instance).all(axis=1))
    i = ind[0][0]
    return EXPLAINERS[i]


def explain_instance_b(instance):
    ind = np.where((NUMERICAL_NP_ARRAY == instance).all(axis=1))
    i = ind[0][0]
    return EXPLAINERS[::-1][i]


def test_validate_input():
    """
    Tests :func:`fatf.transparency.models.submodular_pick._validate_input`.
    """
    explain_instance = lambda x: x + 1  # noqa: E731

    msg = 'The input data set must be a 2-dimensional array.'
    with pytest.raises(IncorrectShapeError) as exin:
        ftms._validate_input(
            np.array([0, ]), None, None, None)  # yapf: disable
    assert str(exin.value) == msg

    msg = ('The input data set must only contain base types (strings and '
           'numbers).')
    with pytest.raises(ValueError) as exin:
        ftms._validate_input(NOT_BASE_NP_ARRAY, None, None, None)
    assert str(exin.value) == msg

    msg = 'sample_size must be an integer.'
    with pytest.raises(TypeError) as exin:
        ftms._validate_input(NUMERICAL_NP_ARRAY, explain_instance, 'int', None)
    assert str(exin.value) == msg

    msg = 'sample_size must be a non-negative integer.'
    with pytest.raises(ValueError) as exin:
        ftms._validate_input(NUMERICAL_NP_ARRAY, explain_instance, -1, None)
    assert str(exin.value) == msg

    msg = 'explanations_number must be an integer.'
    with pytest.raises(TypeError) as exin:
        ftms._validate_input(NUMERICAL_NP_ARRAY, explain_instance, 1, 'a')
    assert str(exin.value) == msg

    msg = 'explanations_number must be a non-negative integer.'
    with pytest.raises(ValueError) as exin:
        ftms._validate_input(NUMERICAL_NP_ARRAY, explain_instance, 1, -1)
    assert str(exin.value) == msg

    msg = ('The explain_instance should be a Python callable '
           '(function or method).')
    with pytest.raises(TypeError) as exin:
        ftms._validate_input(NUMERICAL_NP_ARRAY, None, 1, 1)
    assert str(exin.value) == msg

    msg = ('The explain_instance callable must accept '
           'exactly one required parameter.')
    cal = lambda x, y: x + y  # noqa: E731
    with pytest.raises(RuntimeError) as exin:
        ftms._validate_input(NUMERICAL_NP_ARRAY, cal, 1, 1)
    assert str(exin.value) == msg

    msg = ('The number of explanations cannot be larger than '
           'the number of samples.')
    with pytest.raises(ValueError) as exin:
        ftms._validate_input(NUMERICAL_NP_ARRAY, explain_instance, 1, 2)
    assert str(exin.value) == msg

    # All OK
    assert ftms._validate_input(NUMERICAL_NP_ARRAY, explain_instance, 1, 1)
    assert ftms._validate_input(
        CATEGORICAL_STRUCT_ARRAY, explain_instance, 0, 0)  # yapf: disable
    assert ftms._validate_input(NUMERICAL_NP_ARRAY, explain_instance, 0, 1)


def test_submodular_pick():
    """Tests :func:`fatf.transparency.models.submodular_pick`."""
    fatf.setup_random_seed()

    explanations, explanation_ind = ftms.submodular_pick(
        NUMERICAL_NP_ARRAY, explain_instance_a, explanations_number=2)
    assert explanation_ind == [0, 2]
    assert explanations == [EXPLAINERS[0], EXPLAINERS[2]]

    explanations, explanation_ind = ftms.submodular_pick(
        NUMERICAL_NP_ARRAY, explain_instance_b, explanations_number=2)
    assert explanation_ind == [0, 1]
    assert explanations == [EXPLAINERS[3], EXPLAINERS[2]]

    msg = ('sample_size is larger than the number of samples in the data set. '
           'The whole dataset will be used.')
    with pytest.warns(UserWarning) as warning:
        explanations, explanation_ind = ftms.submodular_pick(
            NUMERICAL_NP_ARRAY,
            explain_instance_a,
            sample_size=100,
            explanations_number=1)
    assert len(warning) == 1
    assert str(warning[0].message) == msg
    assert explanation_ind == [0]
    assert explanations == [EXPLAINERS[0]]

    explanations, explanation_ind = ftms.submodular_pick(
        NUMERICAL_NP_ARRAY,
        explain_instance_a,
        sample_size=1,
        explanations_number=1)
    assert explanation_ind == [1]
    assert explanations == [EXPLAINERS[1]]

    explanations, explanation_ind = ftms.submodular_pick(
        NUMERICAL_NP_ARRAY,
        explain_instance_a,
        sample_size=0,
        explanations_number=0)
    assert explanation_ind == [0, 2, 1, 3]
    assert explanations == [
        EXPLAINERS[0], EXPLAINERS[2], EXPLAINERS[1], EXPLAINERS[3]
    ]

    explanations, explanation_ind = ftms.submodular_pick(
        NUMERICAL_NP_ARRAY,
        explain_instance_a,
        sample_size=2,
        explanations_number=0)
    assert explanation_ind == [3, 1]
    assert explanations == [EXPLAINERS[3], EXPLAINERS[1]]

    msg = ('The number of explanations cannot be larger than '
           'the number of instances (rows) in the data set.')
    with pytest.warns(UserWarning) as warning:
        explanations, explanation_ind = ftms.submodular_pick(
            NUMERICAL_NP_ARRAY, explain_instance_a, 0, 222)
    assert len(warning) == 1
    assert str(warning[0].message) == msg
    assert explanation_ind == [0, 2, 1, 3]
    assert explanations == [
        EXPLAINERS[0], EXPLAINERS[2], EXPLAINERS[1], EXPLAINERS[3]
    ]
