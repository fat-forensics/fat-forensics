"""
Tests model_comparison functions.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
# License: new BSD

from typing import Dict, Union

import pytest

import numpy as np

import fatf

from fatf.exceptions import IncompatibleModelError, IncorrectShapeError

NUMERICAL_NP_ARRAY = np.array([[0, 0, 0.08, 0.69], [1, 0, 0.03, 0.29],
                               [0, 1, 0.99, 0.82], [2, 1, 0.73, 0.48],
                               [1, 0, 0.36, 0.89], [0, 1, 0.07, 0.21]])
NUMERICAL_NP_ARRAY_TARGET = np.array([2, 0, 1, 1, 0, 2])


def test_input_is_valid():
    """
    Tests :func:`fatf.utils.transparency.model_comparison._input_is_valid`.
    """
    # TODO: test
    return True

def test_local_fidelity_score():
    """
    Tests :func:`fatf.utils.transparency.model_comparison.
    local_fidelity_score`.
    """
    # TODO: test
    return True