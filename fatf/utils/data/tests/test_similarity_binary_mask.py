"""
Functions for testing similarity binary mask function.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
# License: new BSD

import pytest

import numpy as np

import fatf

from fatf.exceptions import IncorrectShapeError

import fatf.utils.data.similarity_binary_mask as fuds


NUMERICAL_NP_ARRAY = np.array([
    [0, 0, 1., 0.],
    [1, 0, 2., 4.],
    [0, 1, 3., 0.],
    [2, 1, 2., 1.],
    [1, 0, 1., 2.],
    [0, 1, 1., 0.]])
NUMERICAL_STRUCT_ARRAY = np.array(
    [(0, 0, 1., 0.),
     (1, 0, 2., 4.),
     (0, 1, 3., 0.),
     (2, 1, 2., 1.),
     (1, 0, 1., 2.),
     (0, 1, 1., 0.)],
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

NUMERICAL_NP_BINARY = np.array([
    [1, 1, 1, 1],
    [0, 1, 0, 0],
    [1, 0, 0, 1],
    [0, 0, 0, 0],
    [0, 1, 1, 0],
    [1, 0, 1, 1]])
NUMERICAL_STRUCT_BINARY = np.array(
    [(1, 1, 1, 1),
     (0, 1, 0, 0),
     (1, 0, 0, 1),
     (0, 0, 0, 0),
     (0, 1, 1, 0),
     (1, 0, 1, 1)],
    dtype=[('a', 'i'), ('b', 'i'), ('c', 'i'), ('d', 'i')])


def test_validate_input():
    """
    Tests :func:`fatf.utils.data.similarity_binary_mask._validate_input`.
    """
    return True

def test_similarity_binary_mask():
    """
    Tests :func:`fatf.utils.data.similarity_binary_mask.similarity_binary_mask`.
    """
    binary = fuds.similarity_binary_mask(NUMERICAL_NP_ARRAY,
                                         NUMERICAL_NP_ARRAY[0])
    assert np.array_equal(binary, NUMERICAL_NP_BINARY)
    binary = fuds.similarity_binary_mask(NUMERICAL_STRUCT_ARRAY,
                                         NUMERICAL_STRUCT_ARRAY[0])
    assert np.array_equal(binary, NUMERICAL_STRUCT_BINARY)
