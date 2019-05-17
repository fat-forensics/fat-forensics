"""
Tests Blimey class.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
# License: new BSD

import pytest

import numpy as np

import fatf

import fatf.transparency.models.blimey as ftmb
import fatf.utils.models as fum
import fatf.utils.data.augmentation as fuda
import fatf.utils.data.discretization as fudd

NUMERICAL_NP_ARRAY = np.array([
    [0, 0, 0.08, 0.69],
    [1, 0, 0.03, 0.29],
    [0, 1, 0.99, 0.82],
    [0, 1, 0.07, 0.21]])

NUMERICAL_NP_ARRAY_TARGET = np.array([2, 0, 1, 1, 0, 2])


def test_input_is_valid():
    """
    Tests :func:`fatf.transparency.models.blimey._is_input_valid`.
    """


class TestBlimey():
    """
    Tests :class:`fatf.transparency.models.blimey.Blimey`.
    """
    knn_numerical = fum.KNN()
    knn_numerical.fit(NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY_TARGET)

    numerical_blimey = 

