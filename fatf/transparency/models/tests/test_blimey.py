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
import fatf.transparency.models.submodular_pick as ftmsp

from sklearn.linear_model import Ridge

NUMERICAL_NP_ARRAY_TARGET = np.array([2, 0, 1, 1, 0, 2])
NUMERICAL_NP_ARRAY = np.array([
    [0, 0, 0.08, 0.69],
    [1, 0, 0.03, 0.29],
    [0, 1, 0.99, 0.82],
    [2, 1, 0.73, 0.48],
    [1, 0, 0.36, 0.89],
    [0, 1, 0.07, 0.21]])


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

    numerical_blimey = ftmb.blimey(
        dataset=NUMERICAL_NP_ARRAY,
        augmentor=fuda.NormalSampling,
        discretizer=fudd.QuartileDiscretizer,
        explainer=ftmsp.SKLearnExplainer,
        global_model=knn_numerical,
        local_model=Ridge,
        class_names=['Class A', 'Class B', 'Class C'],
        feature_names=['A', 'B', 'C', 'D'])

    exp = numerical_blimey.explain_instance(NUMERICAL_NP_ARRAY[0])

